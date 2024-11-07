import os
import json
import bz2
import time
import logging
from typing import Dict, Any

import torch
from PIL import Image

class ImageCompression:
    def unzip_image(self, zip_path: str, output_path: str) -> None:
        """解压图片文件
        
        Args:
            zip_path: 压缩文件路径
            output_path: 输出图片路径
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)  # 添加这行来创建输出目录
            
            print(f"\n开始解压图片: {zip_path}")
            start_time = time.time()
            
            # 读取压缩数据
            with open(zip_path, 'rb') as f:
                data = f.read()
            metadata = json.loads(bz2.decompress(data).decode())
            
            # 解码图片
            img = self.decode_image(metadata)
            
            # 保存图片
            print(f"\n保存解压后的图片: {output_path}")
            img.save(output_path)
            
            end_time = time.time()
            print(f"\n解压完成! 总用时: {end_time - start_time:.2f} 秒")
            
        except Exception as e:
            logging.error(f"Error unzipping image: {str(e)}")
            raise 

    def tokens_to_image(self, tokens: torch.Tensor, metadata: Dict[str, Any]) -> Image.Image:
        try:
            # ... 现有代码 ...
            
            # 创建图片
            img = Image.fromarray(img_array)
            print(f"图片重建完成，尺寸: {img.size}")
            
            # 验证图片是否有效
            if img.size != tuple(metadata['size']):
                raise ValueError(f"重建图片尺寸不匹配: 期望 {metadata['size']}, 实际 {img.size}")
                
            return img
                
        except Exception as e:
            logging.error(f"Error converting tokens to image: {str(e)}")
            if 'img_array' in locals():
                logging.error(f"Image array shape: {img_array.shape}")
            raise