import torch

# Creating a tensor

# scalar
scalar = torch.tensor(7)
print(f"{scalar=}")
print(f"{scalar.ndim=}")
print(f"{scalar.item()=}")

#vector
vector = torch.tensor([7, 7])
print(f"{vector.ndim=}")
print(f"{vector.shape=}")

# MATRIX
MATRIX = torch.tensor([[7,8],[9,10]])
print(f"{MATRIX.ndim=}")
print(f"{MATRIX.shape=}")

#TENSOR
TENSOR = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[0,0,0],[1,1,1],[2,2,2]]])
print(f"{TENSOR.ndim=}")
print(f"{TENSOR.shape=}")

# Random Tensor of size (3, 4)
random_tensor = torch.rand(3,4,5,6)
print(f"{random_tensor=}")
print(f"{random_tensor.ndim=}")
print(f"{random_tensor.shape=}")

f32 = torch.tensor([3.0, 6.0, 9.0],
        dtype=None,
        device=None,
        requires_grad=False)
print(f"{f32.device=}")

f32_1 = torch.tensor([9, 10, 11])
print(f"{f32 * f32_1=}")

t1 = torch.rand(1,3,3)
print(f"{t1=}")
t2 = torch.rand(1,3,3)
print(f"{t2=}")
print(f"{torch.matmul(t1, t2)=}")
print(f"{(t1 @ t2)=}")

t1 = torch.rand(3,2)
t2 = torch.rand(3,2)
# Note can't multiply, as "inner dimensions" must match
# so TRANSPOSE one of the pair of tensors
print(f"{(t1 @ t2.T) =}")

print(f"{t1.squeeze()=}")
print(f"{t1.shape=}")

t1s = t1.squeeze()
print(f"{t1s=}")

t3 = torch.tensor([1,2,3,4])
print(f"{t3=}")
t4 = torch.stack([t3, t3, t3])
print(f"{t4=}")

t5 = torch.tensor([1,2,3,4])
print(f"{t5=}")
t6 = t5.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
print(f"{t6=}")
t7 = t6.squeeze()
print(f"{t7=}")
