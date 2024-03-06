import numpy as np
import torch 

""" numpy to tensor """
array = np.arange(1.0, 8.0)

tensor = torch.from_numpy(array).type(torch.float32)
print(f"{type(array)=}, {type(tensor)=}")
print(f"{array=}, {tensor=}")
print("Add 1 to np.array elements")
array += 1
print(f"{array=}, {tensor=}")

print("\ncheck out what happens when you don't cast the array to float32!")
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) #don't cast to float32
print(f"{array=}, {tensor=}")
print("add 1 to np.array elements")
array += 1
print(f"{array=}, {tensor=}")

""" Going from torch =>  numpy """
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(f"{tensor=}, {numpy_tensor=}")
print("add 1 to tensor")
tensor += 1
print(f"{tensor=}, {numpy_tensor=}")

""" Going from numpy to torch """
tensor = torch.tensor(numpy_tensor)
print(tensor)


