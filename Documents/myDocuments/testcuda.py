import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


if torch.cuda.is_available():
    print('yes')
else:
    print('NO')
    

# 检查是否有可用的 GPU
print(torch.cuda.is_available()) # FALSE

# 检查 CUDA 版本
print(torch.version.cuda) #NO

# 列出可用的设备
print(torch.cuda.device_count()) #0


# 检查 PyTorch 版本
print(torch.__version__) #1.12.1

# 检查 CUDA 版本
print(torch.version.cuda)#NO

print(torch.__config__.show())  # 显示 PyTorch 的编译配置
