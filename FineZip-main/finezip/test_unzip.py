import os
import traceback
from image_compression import create_image_zip_model

def test_unzip_only():
    """仅测试解压功能"""
    try:
        # 设置路径
        zip_path = "test_output/compressed.gpz"
        output_path = "test_output/decompressed_test.png"
        
        if not os.path.exists(zip_path):
            print(f"错误: 找不到压缩文件 {zip_path}")
            return
            
        print("正在创建模型实例...")
        model = create_image_zip_model("meta-llama/Llama-3.1-8B")
        
        # 仅执行解压
        print(f"\n开始解压测试...")
        model.unzip_image(zip_path, output_path)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_unzip_only() 