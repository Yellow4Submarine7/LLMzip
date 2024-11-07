import os
import traceback
from image_compression import create_image_zip_model
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def test_image_compression(image_path: str):
    """测试图片压缩功能
    
    Args:
        image_path: 测试图片路径
    """
    # 检查输入图片是否存在
    if not os.path.exists(image_path):
        print(f"错误: 找不到图片文件 {image_path}")
        return
        
    # 创建输出目录
    os.makedirs("test_output", exist_ok=True)
    
    # 设置输出路径
    zip_path = os.path.join("test_output", "compressed.gpz")
    output_path = os.path.join("test_output", "decompressed.png")
    
    try:
        # 创建模型实例
        print("正在创建模型实例...")
        model = create_image_zip_model("meta-llama/Llama-3.1-8B")
        
        # 压缩图片
        print(f"\n开始压缩图片: {image_path}")
        compression_ratio = model.zip_image(image_path, zip_path)
        print(f"压缩完成! 压缩比: {compression_ratio:.4f}")
        
        # 解压图片
        print("\n开始解压图片...")
        model.unzip_image(zip_path, output_path)
        print(f"解压完成! 输出路径: {output_path}")
        
        # 计算图片质量指标
        original_img = np.array(Image.open(image_path).convert('RGB'))
        decompressed_img = np.array(Image.open(output_path).convert('RGB'))
        
        # 计算PSNR (峰值信噪比)
        psnr = peak_signal_noise_ratio(original_img, decompressed_img)
        
        # 计算SSIM (结构相似性)
        ssim = structural_similarity(original_img, decompressed_img, channel_axis=2)
        
        print("\n图片质量评估:")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.4f}")
        
        # 输出文件大小信息
        original_size = os.path.getsize(image_path)
        compressed_size = os.path.getsize(zip_path)
        print("\n文件大小比较:")
        print(f"原始大小: {original_size/1024:.2f} KB")
        print(f"压缩大小: {compressed_size/1024:.2f} KB")
        print(f"压缩率: {compressed_size/original_size*100:.2f}%")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    # 使用PNG格式的测试图片
    test_image = "test_images/test.png"
    print(f"当前工作目录: {os.getcwd()}")
    print(f"测试图片路径: {os.path.abspath(test_image)}")
    test_image_compression(test_image)