import torch
import torch.nn as nn
import torch.nn.functional as F

inputs = torch.tensor([[1.0, 2.0, 3.0, 2.5],
                       [2.0, 5.0, -1.0, 2.0],
                       [-1.5, 2.7, 3.3, -0.8]])

weights1 = torch.tensor([[0.2, 0.8, -0.5, 1.0],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]])

biases1 = torch.tensor([[2.0, 3.0, 0.5]])

weights2 = torch.tensor([[0.1, -0.14, 0.5],
                         [-0.5, 0.12, -0.33],
                         [-0.44, 0.73, -0.13]])

biases2 = torch.tensor([[-1, 2, -0.5]])

my_layer1 = nn.Linear(4, 3)
my_layer2 = nn.Linear(3, 3)

my_layer1.weight.data = weights1
my_layer1.bias.data = biases1

my_layer2.weight.data = weights2
my_layer2.bias.data = biases2

output1 = my_layer1(inputs)
relu_output1 = F.relu(output1)
output2 = my_layer2(relu_output1)
relu_output2 = F.relu(output2)

# No activation function
print(my_layer2(output1))
# With activation function
print(relu_output2)