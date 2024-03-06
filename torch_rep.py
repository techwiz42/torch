import torch

#create two random tensors
tA = torch.rand(3,3)
tB = torch.rand(3,3)
print(tA)
print(tB)
print(f"{(tA == tB)=}")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
tC = torch.rand(3,3)
tD = torch.rand(3,3)

print(tC)
print(tD)
print(tC == tD)

torch.manual_seed(RANDOM_SEED)
tC = torch.rand(3,3)
torch.manual_seed(RANDOM_SEED)
tD = torch.rand(3,3)
print(tC)
print(tD)
print(tC == tD)
