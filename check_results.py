import os
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

def check_compression_results():
    print("\n=== 检查压缩结果 ===")
    
    # 获取正确的根目录路径
    project_root = os.path.dirname(os.path.abspath(__file__))  # F:\LLMascompressor
    
    # 检查各个目录
    directories = {
        'logs': os.path.join(project_root, "FineZip-main", "logs"),
        'zipped': os.path.join(project_root, "zipped"),  # F:\LLMascompressor\zipped
        'plots': os.path.join(project_root, "plots")     # F:\LLMascompressor\plots
    }
    
    for dir_name, dir_path in directories.items():
        print(f"\n=== 检查{dir_name}目录 ===")
        if not os.path.exists(dir_path):
            print(f"目录不存在: {dir_path}")
            continue
            
        if dir_name == 'logs':
            log_files = os.listdir(dir_path)
            print(f"\n找到 {len(log_files)} 个日志文件:")
            for log in log_files:
                log_path = os.path.join(dir_path, log)
                size = os.path.getsize(log_path)
                print(f"文件: {log}, 大小: {size/1024:.2f}KB")
        
        elif dir_name == 'zipped':
            zipped_files = os.listdir(dir_path)
            model_data = {}
            
            print(f"\n找到 {len(zipped_files)} 个文件:")
            for zipped in zipped_files:
                file_path = os.path.join(dir_path, zipped)
                if zipped.endswith('.gpz'):
                    size = os.path.getsize(file_path)
                    print(f"压缩文件: {zipped}, 大小: {size/1024/1024:.2f}MB")
                
                if zipped.endswith('.txt'):
                    model_name = zipped.split('_256_16.txt')[0]
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        ranks = [int(line.strip()) for line in lines[:-1]]
                        compression_time = float(lines[-1])
                        
                        model_data[model_name] = {
                            'ranks': ranks,
                            'time': compression_time
                        }
                        print(f"\nRanks文件: {zipped}")
                        print(f"记录数量: {len(ranks)}")
                        print(f"压缩时间: {compression_time:.2f}秒")
            
            # 如果有数据就生成图表
            if model_data:
                print("\n=== 生成评估图表 ===")
                fig, axs = plt.subplots(1, 3, figsize=(30, 6))
                
                axs[0].set_title('Percentage of Ranks Between 0-15')
                axs[1].set_title('Percentage of Ranks at 0')
                axs[2].set_title('Rank Distribution (Histogram)')
                
                colors = cycle(['blue', 'green', 'red'])
                
                for model_name, data in model_data.items():
                    color = next(colors)
                    ranks = data['ranks']
                    
                    # 计算0-15的比例
                    ranks_0_15 = len([r for r in ranks if 0 <= r <= 15]) / len(ranks) * 100
                    axs[0].bar(model_name, ranks_0_15, color=color)
                    
                    # 计算0的比例
                    ranks_0 = len([r for r in ranks if r == 0]) / len(ranks) * 100
                    axs[1].bar(model_name, ranks_0, color=color)
                    
                    # 绘制rank分布直方图
                    axs[2].hist(ranks, bins=100, alpha=0.5, label=model_name, color=color)
                
                for ax in axs[:2]:
                    ax.set_ylabel('Percentage')
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                axs[2].set_xlabel('Rank')
                axs[2].set_ylabel('Frequency')
                axs[2].set_yscale('log')
                axs[2].legend()
                
                plt.tight_layout()
                
                # 确保plots目录存在
                os.makedirs(directories['plots'], exist_ok=True)
                plt.savefig(os.path.join(directories['plots'], 'compression_results_analysis.png'))
                print("评估图表已保存为: plots/compression_results_analysis.png")

if __name__ == "__main__":
    check_compression_results() 