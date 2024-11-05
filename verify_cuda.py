import torch
import torchvision
import transformers
import peft

print("=== CUDA 验证 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"Torchvision版本: {torchvision.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"当前设备: {torch.cuda.get_device_name(0)}")
print("\n=== 其他库版本 ===")
print(f"Transformers版本: {transformers.__version__}")
print(f"Peft版本: {peft.__version__}") 