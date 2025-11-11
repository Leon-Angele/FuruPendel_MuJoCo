import torch

# 1. Prüfen, ob CUDA generell verfügbar ist (gibt True oder False aus)
cuda_verfügbar = torch.cuda.is_available()
print(f"CUDA verfügbar: {cuda_verfügbar}")

if cuda_verfügbar:
    # 2. Wenn verfügbar, den Namen der GPU ausgeben
    print(f"Gerätename: {torch.cuda.get_device_name(0)}")
    
    # 3. Prüfen, welche CUDA-Version PyTorch verwendet
    print(f"PyTorch CUDA Version: {torch.version.cuda}")