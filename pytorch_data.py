import torch
import numpy as np


data = [[1,2],[5,6],[3,4]]

x_data = torch.tensor(data)

x_rand = torch.rand_like(x_data, dtype=torch.float)



shape = (2, 4, )

random_tensor = torch.rand(shape)

one_tensor = torch.ones(shape)

zero_tensor = torch.zeros(shape)


print(f"random tensor: \n {one_tensor} \n")

tensor = torch.ones(4, 4)
"""print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])"""
tensor[:,1] = 0
print(tensor)


y1 = tensor @ tensor.T
y2 = tensor.matmul
print(y1)
