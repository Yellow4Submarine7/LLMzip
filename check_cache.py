from transformers import AutoModel
from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
import os

def check_model_cache():
    # 打印默认缓存目录
    print("\n=== 默认缓存位置 ===")
    cache_dir = HUGGINGFACE_HUB_CACHE
    print(f"缓存根目录: {cache_dir}")
    
    try:
        # 扫描缓存内容
        cache_info = scan_cache_dir()
        
        print("\n=== 已下载的模型 ===")
        if not cache_info.repos:
            print("没有找到已缓存的模型")
            return
            
        for repo in cache_info.repos:
            print(f"\n模型名称: {repo.repo_id}")
            print(f"占用空间: {repo.size_on_disk / (1024*1024*1024):.2f} GB")
            print(f"具体位置: {repo.repo_path}")
            
            # 显示模型文件
            print("文件列表:")
            for revision in repo.revisions:
                for file in revision.files:
                    print(f"  - {file.file_name} ({file.size_on_disk / (1024*1024):.2f} MB)")
    
    except Exception as e:
        print(f"扫描缓存时出错: {str(e)}")
        print(f"请检查缓存目录是否存在: {cache_dir}")

if __name__ == "__main__":
    check_model_cache() 