import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

inputs = torch.tensor([[-2., -1., 0.]])

output = F.softmax(inputs, dim=1)
print(output)