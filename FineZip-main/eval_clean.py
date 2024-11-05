import os

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 第一步：只测试微调模型
    finetuned_models = {
        "10mb_qlora/4bit/meta-llama-Llama-3.1-8B-enwik10mb_256_r8": "meta-llama/Llama-3.1-8B"
    }
    original_model_names = []  # 先不加载原始模型
    
    dataset_path = os.path.join(project_root, "datasets", "enwik10mb.txt")
    
    print("项目根目录:", project_root)
    print("数据集路径:", dataset_path)
    print("数据集是否存在:", os.path.exists(dataset_path))
    
    memory_eval(
        finetuned_models=finetuned_models, 
        original_model_names=original_model_names,  # 空列表
        context_sizes=[256], 
        batch_size=16,
        file_path=dataset_path,
        save_name="compression_eval_finetuned"
    )

    # 第二步（需要单独运行）：测试原始模型
    # finetuned_models = {}
    # original_model_names = ["meta-llama/Llama-3.1-8B"]
    # memory_eval(
    #     finetuned_models=finetuned_models,
    #     original_model_names=original_model_names,
    #     context_sizes=[256],
    #     batch_size=16,
    #     file_path=dataset_path,
    #     save_name="compression_eval_original"
    # ) 