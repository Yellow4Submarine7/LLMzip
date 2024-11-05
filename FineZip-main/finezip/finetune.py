import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pprint
from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from utils.finetune_utils import CastOutputToFloat, print_trainable_parameters
from huggingface_hub import login
from transformers import BitsAndBytesConfig

def finetune(model, save_path, dataset_path="datasets/enwik10mb.txt", block_size=128, epochs=10, r=8, learning_rate=1e-4, batch_size=4):
    print(f"当前工作目录: {os.getcwd()}")
    print(f"查找数据集: {os.path.abspath(dataset_path)}")
    if os.path.exists(dataset_path):
        print("✓ 数据集文件存在")
    else:
        print("✗ 数据集文件不存在")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model, 
        device_map={"":torch.cuda.current_device()},
        quantization_config=quant_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model)

    print("Model loaded")

    for param in loaded_model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    loaded_model.gradient_checkpointing_enable()  
    loaded_model.enable_input_require_grads()

    loaded_model.lm_head = CastOutputToFloat(loaded_model.lm_head)    

    config = LoraConfig(
        r=r,
        lora_alpha=32, 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    loaded_model = get_peft_model(loaded_model, config)
    print_trainable_parameters(loaded_model)

    loaded_model.to(torch.cuda.current_device())

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    print("Dataset loaded")

    # 为每个模型创建独立的输出目录
    model_output_dir = os.path.join("output", model.replace('/', '-'))
    os.makedirs(model_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size, 
        gradient_accumulation_steps=8,
        max_steps=epochs, 
        learning_rate=learning_rate, 
        fp16=True,
        logging_steps=1,
        output_dir=model_output_dir,  # 使用模型特定的输出目录
        warmup_steps=500,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # 添加更多日志记录选项
        save_strategy="steps",
        save_steps=50,  # 每50步保存一次
        logging_dir=os.path.join(model_output_dir, "logs"),  # 日志目录
        logging_first_step=True,
    )

    trainer = Trainer(
        model=loaded_model, 
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator
    )

    loaded_model.config.use_cache = False  

    trainer.train()
    trainer.save_model(save_path + f"_{epochs}_r{r}") 

    del loaded_model
    del trainer
    torch.cuda.empty_cache()

    print("Finished finetuning")

if __name__ == "__main__":
    login("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") 

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "datasets", "enwik10mb.txt")
    
    save_dir = os.path.join(project_root, "finetuned_models", "10mb_qlora", "4bit")
    os.makedirs(save_dir, exist_ok=True)

    finetune_list = [
        "meta-llama/Llama-3.1-8B"
    ]
    epoch_list = [256]

    for model in finetune_list:
        for e in epoch_list:
            save_path = os.path.join(save_dir, f"{model.replace('/', '-')}-enwik10mb")
            finetune(
                model=model,
                save_path=save_path,
                dataset_path=dataset_path,
                block_size=64,
                epochs=e,
                r=8,
                learning_rate=1e-4,
                batch_size=4
            )
