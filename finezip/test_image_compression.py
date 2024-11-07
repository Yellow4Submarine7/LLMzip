import os

def test_image_compression(image_path: str):
    """测试图片压缩功能"""
    # 检查输入图片是否存在
    if not os.path.exists(image_path):
        print(f"错误: 找不到图片文件 {image_path}")
        return
        
    # 创建输出目录（使用绝对路径）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出路径（使用绝对路径）
    zip_path = os.path.join(output_dir, "compressed.gpz")
    output_path = os.path.join(output_dir, "decompressed.png") 