import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
#print(f"Tensor from data: \n {x_data} \n")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
#print(f"Tensor from numpy: \n {x_np} \n")

x_ones = torch.ones_like(x_data)
#print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
#print(f"Random Tensor: \n {x_rand} \n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#print(f"Random Tensor: \n {rand_tensor} \n")
#print(f"Ones Tensor: \n {ones_tensor} \n")
#print(f"Zeros Tensor: \n {zeros_tensor}")


tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)