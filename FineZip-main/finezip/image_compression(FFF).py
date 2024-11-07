import os
import sys
import json
import time
import logging
import io
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import torch
import bz2
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 导入父类ZipModel
from eval_clean import ZipModel

class ImageZipModel(ZipModel):
    """用于图片压缩的ZipModel扩展类"""
    
    # 定义常量
    BLOCK_SIZE = 8  # 8x8像素块
    BLOCK_SEPARATOR = '|'  # 块之间的分隔符，使用单个字符
    
    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int) -> str:
        """将RGB值转换为十六进制格式"""
        return f"{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
        """将十六进制格式转换为RGB值"""
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    
    def __init__(self, model_name: str, tokenizer_name: str, model: Any, 
                 tokenizer: Any, finetuned: bool, context_size: int, batch_size: int):
        """初始化ImageZipModel
        
        Args:
            与ZipModel相同的参数
        """
        super().__init__(model_name, tokenizer_name, model, tokenizer, 
                        finetuned, context_size, batch_size)
        
        # 确保device设置正确
        self.device = next(model.parameters()).device
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志系统"""
        # 创建logs目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(
            logs_dir, 
            f'image_compression_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        # 修改日志配置，移除encoding参数
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        logging.info("ImageZipModel initialized")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Context size: {self.CONTEXT_SIZE}")
        logging.info(f"Batch size: {self.BATCH_SIZE}")

    def image_to_tokens(self, image_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """将图片转换为tokens序列
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            tokens: 转换后的token序列
            metadata: 包含图片信息的元数据
        """
        try:
            # 1. 读取图片为numpy数组
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # 保存元数据
            metadata = {
                'size': img.size,
                'mode': img.mode,
                'shape': img_array.shape
            }
            
            print(f"\n原始图片信息:")
            print(f"尺寸: {metadata['size']}")
            print(f"模式: {metadata['mode']}")
            print(f"数组形状: {metadata['shape']}")
            
            # 2. 将图片分块处理
            height, width = img_array.shape[:2]
            pixel_values = []
            
            print("\n开始图片分块处理...")
            for i in range(0, height, self.BLOCK_SIZE):
                for j in range(0, width, self.BLOCK_SIZE):
                    # 提取8x8块
                    block = img_array[i:min(i+self.BLOCK_SIZE, height), 
                                    j:min(j+self.BLOCK_SIZE, width)]
                    
                    # 将RGB值转换为十六进制表示
                    block_values = []
                    for pixel in block.reshape(-1, 3):
                        r, g, b = pixel
                        hex_color = self.rgb_to_hex(r, g, b)
                        block_values.append(hex_color)
                    
                    # 直接连接像素值，不使用空格
                    block_str = ''.join(block_values)
                    pixel_values.append(block_str)
            
            # 3. 使用分隔符连接所有块
            img_str = self.BLOCK_SEPARATOR.join(pixel_values)
            
            print(f"\n数据转换信息:")
            print(f"块数量: {len(pixel_values)}")
            print(f"字符串长度: {len(img_str)}")
            print(f"示例块: {pixel_values[0][:100]}...")
            
            # 4. 转换为tokens
            tokens = self.text_to_tokens(img_str)
            
            print(f"\ntoken信息:")
            print(f"token序列长度: {len(tokens)}")
            print(f"前10个token: {tokens[:10].tolist()}")
            
            # 添加序列长度检查和处理
            MAX_SEQ_LENGTH = 131072  # 模型的最大序列长度
            
            # 如果token序列太长，进行分块处理
            if len(tokens) > MAX_SEQ_LENGTH:
                print(f"\n警告: token序列长度({len(tokens)})超过模型限制({MAX_SEQ_LENGTH})")
                print("进行分块处理...")
                
                # 将tokens分成多个块
                num_chunks = (len(tokens) + MAX_SEQ_LENGTH - 1) // MAX_SEQ_LENGTH
                token_chunks = []
                for i in range(num_chunks):
                    start_idx = i * MAX_SEQ_LENGTH
                    end_idx = min((i + 1) * MAX_SEQ_LENGTH, len(tokens))
                    token_chunks.append(tokens[start_idx:end_idx])
                
                # 更新metadata以包含分块信息
                metadata['num_chunks'] = num_chunks
                metadata['chunk_sizes'] = [len(chunk) for chunk in token_chunks]
                metadata['token_str'] = img_str  # 保存完整的token字符串
                
                print(f"分块完成，共{num_chunks}个块")
                # 返回第一个块和元数据
                return token_chunks[0], metadata
            
            return tokens, metadata
            
        except Exception as e:
            logging.error(f"Error converting image to tokens: {str(e)}")
            raise

    def tokens_to_image(self, tokens: torch.Tensor, metadata: Dict[str, Any]) -> Image.Image:
        """将tokens序列转换回图片"""
        try:
            # 1. 将tokens转换回字符串
            img_str = self.tokens_to_text(tokens)
            
            print("\n开始图片重建...")
            print(f"字符串长度: {len(img_str)}")
            
            # 2. 分割成像素块
            pixel_blocks = img_str.split(self.BLOCK_SEPARATOR)
            print(f"块数量: {len(pixel_blocks)}")
            
            # 3. 重建图片数组
            height, width = metadata['shape'][:2]
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            block_idx = 0
            for i in range(0, height, self.BLOCK_SIZE):
                for j in range(0, width, self.BLOCK_SIZE):
                    if block_idx >= len(pixel_blocks):
                        break
                        
                    # 解析当前块的像素值
                    try:
                        block_str = pixel_blocks[block_idx].strip()
                        pixel_values = []
                        
                        # 每6个字符代表一个像素的RGB值
                        for k in range(0, len(block_str), 6):
                            hex_color = block_str[k:k+6]
                            if hex_color:  # 确保不是空字符串
                                r, g, b = self.hex_to_rgb(hex_color)
                                pixel_values.append((r, g, b))
                        
                        # 填充图片数组
                        for k, (r, g, b) in enumerate(pixel_values):
                            row = i + k // self.BLOCK_SIZE
                            col = j + k % self.BLOCK_SIZE
                            if row < height and col < width:
                                img_array[row, col] = [r, g, b]
                                
                    except Exception as e:
                        logging.error(f"Error processing block {block_idx}: {str(e)}")
                        logging.error(f"Block content: {block_str[:100]}...")
                        
                    block_idx += 1
            
            # 4. 创建图片
            img = Image.fromarray(img_array)
            print(f"图片重建完成，尺寸: {img.size}")
            
            return img
            
        except Exception as e:
            logging.error(f"Error converting tokens to image: {str(e)}")
            raise

    def encode_image(self, image_path: str) -> Dict[str, Any]:
        """压缩图片
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            包含压缩数据和元数据的字典
        """
        logging.info(f"Starting image encoding: {image_path}")
        start_time = time.time()
        
        try:
            # 转换图片为tokens
            tokens, metadata = self.image_to_tokens(image_path)
            
            # 检查是否需要分块处理
            if 'num_chunks' in metadata:
                print(f"\n开始处理{metadata['num_chunks']}个数据块...")
                # 分块处理
                encoded_chunks = []
                for i in range(metadata['num_chunks']):
                    print(f"处理第 {i+1}/{metadata['num_chunks']} 块...")
                    if i == 0:
                        # 第一个块已经在tokens中
                        chunk_tokens = tokens
                    else:
                        # 重新获取其他块
                        chunk_start = i * 131072
                        chunk_end = min((i + 1) * 131072, sum(metadata['chunk_sizes']))
                        chunk_tokens = self.text_to_tokens(metadata['token_str'][chunk_start:chunk_end])
                    
                    # 确保chunk_tokens是正确的张量格式
                    if not isinstance(chunk_tokens, torch.Tensor):
                        chunk_tokens = torch.tensor(chunk_tokens, device=self.device)
                    
                    # 编码当前块
                    encoded_chunk = self.encode_tokens(chunk_tokens)
                    encoded_chunks.append(encoded_chunk.tolist())
                    print(f"第 {i+1} 块处理完成")
                
                metadata['encoded_chunks'] = encoded_chunks
                print("所有数据块处理完成")
            else:
                # 单块处理
                encoded = self.encode_tokens(tokens)
                metadata['encoded'] = encoded.tolist()
            
            end_time = time.time()
            logging.info(f"Image encoding completed in {end_time - start_time:.2f} seconds")
            return metadata
            
        except Exception as e:
            logging.error(f"Error encoding image: {str(e)}")
            raise

    def decode_image(self, metadata: Dict[str, Any]) -> Image.Image:
        """解压图片
        
        Args:
            metadata: 包含压缩数据和元数据的字典
            
        Returns:
            解压后的图片对象
        """
        logging.info("Starting image decoding")
        start_time = time.time()
        
        try:
            # 提取编码数据
            encoded = torch.tensor(metadata['encoded'])
            
            # 使用父类的decode_tokens方法解码
            tokens = self.decode_tokens(encoded)
            
            # 转换回图片
            img = self.tokens_to_image(tokens, metadata)
            
            end_time = time.time()
            logging.info(f"Image decoding completed in {end_time - start_time:.2f} seconds")
            return img
            
        except Exception as e:
            logging.error(f"Error decoding image: {str(e)}")
            raise

    def zip_image(self, image_path: str, zip_path: str) -> float:
        """压缩图片文件
        
        Args:
            image_path: 输入图片路径
            zip_path: 输出压缩文件路径
            
        Returns:
            压缩比
        """
        start_time = time.time()
        logging.info(f"Starting image compression: {image_path}")
        
        try:
            # 压缩图片
            metadata = self.encode_image(image_path)
            
            # 使用BZ2进行二次压缩
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            compressed_bytes = bz2.compress(metadata_bytes)
            
            # 保存压缩文件
            with open(zip_path, "wb") as f:
                f.write(compressed_bytes)
            
            # 计算压缩统计信息
            end_time = time.time()
            original_size = os.path.getsize(image_path)
            compressed_size = os.path.getsize(zip_path)
            compression_ratio = compressed_size / original_size
            
            # 记录日志
            logging.info(f"Compression completed in {end_time - start_time:.2f} seconds")
            logging.info(f"Original size: {original_size} bytes")
            logging.info(f"Compressed size: {compressed_size} bytes")
            logging.info(f"Compression ratio: {compression_ratio:.4f}")
            
            return compression_ratio
            
        except Exception as e:
            logging.error(f"Error in image compression: {str(e)}")
            raise

    def unzip_image(self, zip_path: str, output_path: str):
        """解压图片文件
        
        Args:
            zip_path: 压缩文件路径
            output_path: 输出图片路径
        """
        start_time = time.time()
        print(f"\n开始图片解压过程: {zip_path}")
        
        try:
            # 读取压缩文件
            print("读取压缩文件...")
            with open(zip_path, "rb") as f:
                compressed_bytes = f.read()
            
            # BZ2解压
            print("执行BZ2解压...")
            metadata_bytes = bz2.decompress(compressed_bytes)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            print("\n元数据信息:")
            print(f"图片尺寸: {metadata['size']}")
            print(f"编码数据长度: {len(metadata['encoded'])}")
            
            # 解码图片
            print("\n开始图片解码...")
            img = self.decode_image(metadata)
            
            # 保存图片
            print(f"\n保存解压后的图片: {output_path}")
            img.save(output_path)
            
            end_time = time.time()
            print(f"\n解压完成! 总用时: {end_time - start_time:.2f} 秒")
            
        except Exception as e:
            print(f"解压过程出错: {str(e)}")
            raise

    @torch.no_grad()
    def get_logits(self, tokens, token_index, past=None):
        """获取模型预测的logits
        
        Args:
            tokens: 输入tokens
            token_index: 当前token的索引
            past: 过去的key/value缓存
            
        Returns:
            logits: 模型预测的logits
            past_key_values: 更新的key/value缓存
        """
        try:
            my_inputs = {}
            my_inputs['input_ids'] = tokens[:, token_index].reshape(-1, 1)
            
            # 添加past_key_values到输入
            if past is not None:
                my_inputs['past_key_values'] = past
                
            # 获取模型输出
            output = self.model(**my_inputs)
            
            # 确保logits是二维的 [batch_size, vocab_size]
            logits = output.logits
            if len(logits.shape) == 3:  # [batch_size, seq_len, vocab_size]
                logits = logits.squeeze(1)  # 移除seq_len维度
                
            return logits, output.past_key_values
            
        except Exception as e:
            logging.error(f"Error in get_logits: {str(e)}")
            logging.error(f"tokens shape: {tokens.shape}")
            logging.error(f"token_index: {token_index}")
            if 'logits' in locals():
                logging.error(f"logits shape: {logits.shape}")
            raise

    def encode_tokens(self, tokens):
        """编码token序列"""
        try:
            print("\n开始token编码过程...")
            
            # 1. 填充处理
            tokens, pad_len = self.pad(tokens, self.tokenizer.eos_token_id)
            tokens = tokens.view(-1, self.CONTEXT_SIZE)
            
            # 2. 批处理
            batches = tokens.shape[0]//self.BATCH_SIZE
            if tokens.shape[0] % self.BATCH_SIZE != 0:
                batches += 1
                
            output_scores = torch.zeros((tokens.shape[0], tokens.shape[1]-1), device=self.device)
            
            print(f"\n开始批处理编码，共 {batches} 批...")
            print(f"每批处理 {self.BATCH_SIZE} 个序列，每个序列长度 {self.CONTEXT_SIZE}")
            total_start_time = time.time()
            
            # 3. 处理每个批次
            for i in range(batches):
                batch_start_time = time.time()
                print(f"\n处理第 {i+1}/{batches} 批...")
                
                cur_tokens = tokens[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE]
                cur_output_scores = torch.zeros((cur_tokens.shape[0], cur_tokens.shape[1]-1), device=self.device)
                past = None
                
                print(f"当前批次大小: {cur_tokens.shape[0]}")
                
                # 处理当前批次的tokens
                for j in range(cur_tokens.shape[1]-1):
                    if j % 100 == 0:  # 每100个token输出一次进度
                        print(f"已处理 {j}/{cur_tokens.shape[1]-1} tokens")
                    cur_output_scores[:, j], past = self.encode_one_batch(cur_tokens, j, past)
                
                output_scores[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE] = cur_output_scores
                
                # 清理内存
                del cur_tokens
                del cur_output_scores
                if past is not None:
                    del past
                torch.cuda.empty_cache()
                
                # 计算时间和进度
                batch_time = time.time() - batch_start_time
                total_time = time.time() - total_start_time
                
                # 计算预计剩余时间
                avg_batch_time = total_time / (i + 1)
                remaining_batches = batches - (i + 1)
                estimated_remaining_time = avg_batch_time * remaining_batches
                
                # 格式化时间显示
                remaining_hours = int(estimated_remaining_time // 3600)
                remaining_minutes = int((estimated_remaining_time % 3600) // 60)
                remaining_seconds = int(estimated_remaining_time % 60)
                
                print(f"第 {i+1}/{batches} 批完成，用时: {batch_time:.2f}秒，总用时: {total_time:.2f}秒")
                print(f"平均每批用时: {avg_batch_time:.2f}秒")
                print(f"预计剩余时间: {remaining_hours}小时 {remaining_minutes}分钟 {remaining_seconds}秒")
                print(f"预计总用时: {(total_time + estimated_remaining_time)/3600:.1f}小时")
            
            # 4. 处理填充
            output_scores = output_scores.flatten().int()
            if pad_len > 0:
                output_scores = output_scores[:-pad_len]
                
            print(f"\n编码完成，总用时: {time.time() - total_start_time:.2f}秒")
            return output_scores
            
        except Exception as e:
            logging.error(f"Error in encode_tokens: {str(e)}")
            if 'tokens' in locals():
                logging.error(f"tokens shape: {tokens.shape}")
            raise

def create_image_zip_model(model_name: str, context_size: int = 512, batch_size: int = 4) -> ImageZipModel:
    """创建ImageZipModel实例的工厂函数
    
    Args:
        model_name: 模型名称
        context_size: 上下文窗口大小
        batch_size: 批处理大小（对于图片压缩，使用较小的batch_size）
        
    Returns:
        ImageZipModel实例
    """
    # 配置4-bit量化
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 创建ImageZipModel实例
    return ImageZipModel(
        model_name=model_name,
        tokenizer_name=model_name,
        model=model,
        tokenizer=tokenizer,
        finetuned=False,
        context_size=context_size,
        batch_size=batch_size
    )

if __name__ == "__main__":
    # 示例用法
    model_name = "meta-llama/Llama-3.1-8B"
    image_path = "test_images/test.png"
    zip_path = "test_output/compressed.gpz"
    output_path = "test_output/decompressed.png"
    
    try:
        # 创建模型实例
        model = create_image_zip_model(model_name)
        
        # 压缩图片
        compression_ratio = model.zip_image(image_path, zip_path)
        print(f"Compression ratio: {compression_ratio:.4f}")
        
        # 解压图片
        model.unzip_image(zip_path, output_path)
        print("Image compression and decompression completed successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")