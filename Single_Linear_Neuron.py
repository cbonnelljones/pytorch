import torch
import torch.nn as nn
import torch.nn.functional as F

inputs = torch.tensor([[1, 2, 3, 2.5]])
#inputs = torch.full((1, 4), 1.)
weights = torch.tensor([[0.2, 0.8, -0.5, 1.0]])
#print(weights)
#weights = torch.full((1, 4), 0.5)
#print(weights)
bias = torch.tensor([[2.0]])

output = nn.Linear(4, 1)
output.weight.data = weights
output.bias.data = bias
print(output(inputs))