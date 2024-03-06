import torch

# Get GPU
gpu_available = torch.cuda.is_available()

# Device agnostic code
device = "cuda" if gpu_available else "cpu"

t1 = torch.tensor([1,2,3])
print(t1, t1.device)
print("move to GPU")
t2 = t1.to(device)
print(t2, t2.device)

# Numpy only runs on cpu
t3 = t2.to("cpu").numpy()
print(t2, t2.device)
print(t3)

