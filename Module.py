import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


weight_combined = []
bias_combined = []

inputs = torch.tensor([[1.0, 2.0, 3.0, 2.5],
                       [2.0, 5.0, -1.0, 2.0],
                       [-1.5, 2.7, 3.3, -0.8]])

weights1 = torch.tensor([[0.2, 0.8, -0.5, 1.0],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]])
weight_combined.append(weights1)

biases1 = torch.tensor([[2.0, 3.0, 0.5]])
bias_combined.append(biases1)

weights2 = torch.tensor([[0.1, -0.14, 0.5],
                         [-0.5, 0.12, -0.33],
                         [-0.44, 0.73, -0.13]])
weight_combined.append(weights2)

biases2 = torch.tensor([[-1., 2., -0.5]])
bias_combined.append(biases2)

target_output = torch.tensor([[1., 0., 0.],
                              [0., 1., 0.],
                              [0., 1., 0.]])

class Net(nn.Module):

    def __init__(self, weights, biases):
        super().__init__() 
        self.fc1 = nn.Linear(4, 3)
        self.fc1.weight.data = weights[0]
        self.fc1.bias.data = biases[0]
        self.fc2 = nn.Linear(3, 3)
        self.fc2.weight.data = weights[1]
        self.fc2.bias.data = biases[1]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=1)

my_net = Net(weights=weight_combined, biases=bias_combined)
output = my_net.forward(inputs)
print(output)

optimizer = optim.Adam(my_net.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()
loss = loss_function(output, target_output)
loss.backward()
optimizer.step()
print(f"Loss: {loss}")