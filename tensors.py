import torch
import numpy as np

data = [[1,2],[3,4]]
#print(f"Direct Data: {data}")
x_data = torch.tensor(data)
#print(f"Tensor from data: {x_data}")

np_array = np.array(data)
#print(f"NP Array: {data}")

x_np = torch.from_numpy(np_array)
#print(f"Tensor from NP: {x_np}")

x_ones = torch.ones_like(x_data) # retains properties of x_data
#print(f"Ones tensor: {x_ones}")
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
#print(f"Random Tensor: {x_rand}")

shape = 2, 3,
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#print(f"Rand: {rand_tensor}")
#print(f"Ones: {ones_tensor}")
#print(f"Zeros: {zeros_tensor}")

tensor = torch.ones(4,4)
tensor[:,1] = 0
#print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)