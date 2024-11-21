import torch
import torch.nn as nn
import torch.nn.functional as F

inputs = torch.tensor([[1, 2, 3, 2.5]])
weights = torch.tensor([[0.2, 0.8, -0.5, 1.0],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]])
bias = torch.tensor([[-2.0, -3.0, -0.5]])

my_layer = nn.Linear(4, 3)
my_layer.weight.data = weights
my_layer.bias.data = bias
output = my_layer(inputs)
relu_output = F.relu(output)
print(output)
print(relu_output)