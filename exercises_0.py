""" 
these exercises are from 
https://www.learnpytorch.io/00_pytorch_fundamentals/#Extra-curriculum
See bottom of page.
"""
import torch

# Create a random tensor with shape (7, 7).
tensor = torch.rand(7,7)
print(f"{tensor=}")
print(f"{tensor.shape=}")

#Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7)
vector = torch.rand(7).type(torch.float32)
print(f"{vector=}")
print(f"{vector.shape=}")
product = torch.matmul(tensor, vector.permute(vector.ndim - 1))
print(f"{product=}")

# Set the random seed to 0 and do exercises 2 & 3 over again.
RANDOM_SEED  = 0
torch.manual_seed(RANDOM_SEED)
tensor = torch.rand(7,7)
vector = torch.rand(7).type(torch.float32)
p2 = tensor @ vector.permute(vector.ndim -1)
print(f"{p2=}")

# Speaking of random seeds, we saw how to set it with torch.manual_seed() 
# but is there a GPU equivalent? 
torch.manual_seed(RANDOM_SEED)
cpu_tensor = torch.rand(7,7)
torch.manual_seed(RANDOM_SEED)
cpu_tensor2 = torch.rand(7,7)
print(f"{(cpu_tensor == cpu_tensor2)=}")
torch.manual_seed(RANDOM_SEED)
gpu_tensor = torch.rand(7,7, device="cuda")
torch.manual_seed(RANDOM_SEED)
gpu_tensor2 = torch.rand(7,7, device="cuda")
print(f"{(gpu_tensor == gpu_tensor2)=}")

# Create two random tensors of shape (2, 3) and send them both to the GPU.
# Set torch.manual_seed(1234)
RANDOM_SEED = 1234
torch.manual_seed(RANDOM_SEED)
gpu_t1 = torch.rand(2, 3, device="cuda")
gpu_t2 = torch.rand(2, 3, device="cuda")

print(f"{gpu_t1=}")
print(f"{gpu_t2=}")

p1 = torch.matmul(gpu_t1, gpu_t2.T)
p2 = torch.matmul(gpu_t1.T, gpu_t2)
print(f"{p1=}, {p1.shape=}")
print(f"{p2=}, {p2.shape=}")

#Find the maximum and minimum values of the output of 7.
print(f"{p1.max()=}, {p1.min()=}")

#Find the maximum and minimum index values of the output of 7.
print(f"{p1.argmax()=}, {p1.argmin()=}")
print(p1)

"""
Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10). Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.
"""
RANDOM_SEED = 7
torch.manual_seed(RANDOM_SEED)
tensor_t = torch.rand(1, 1, 1, 10)
print(f"{tensor_t=}, {tensor_t.shape=}")
tensor_q = tensor_t.squeeze()
print(f"{tensor_q=}, {tensor_q.shape=}")
