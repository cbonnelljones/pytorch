import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import cv2
import os
import matplotlib.pyplot as plt

IMAGE_PATH = "../NeuralNetwork/"
IMAGE_NAME = "pants.jpg"
IMAGE_PROCESS = IMAGE_PATH + IMAGE_NAME


# Read an image
#image_data = cv2.imread(IMAGE_PROCESS, cv2.IMREAD_GRAYSCALE)

# Resize to the same size as the Fashion MNIST images
#image_data = cv2.resize(image_data, (28,28))

# Invert image colors
#image_data = 225 - image_data

#plt.imshow(image_data, cmap="gray")
#plt.show()

classes = {
    0: "T-shirt/top",
    1: "Pants",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Shoe",
    8: "Bag",
    9: "Boot"
}

print_every = 100
batch_size = 15000
training_data = datasets.FashionMNIST(
    root='data', 
    train=True, 
    download=True, 
    transform=ToTensor())

"""
** 
target_transform = features as normalized tensors and the labels as one-hot encoded tensors
https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
"""
trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Net(nn.Module):
    
    def __init__(self):
        super().__init__() 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
    
    def forward(self, inputs):
        inputs = self.flatten(inputs)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        inputs = F.relu(self.fc3(inputs))
        return F.softmax(inputs, dim=1)


#train_features, train_labels = next(iter(trainloader))
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")
#print(f"Labels batch tensor: {train_labels}")
#img = train_features[0].squeeze()
#class_number = train_labels[0].item()
#label = classes[class_number]
#plt.imshow(img, cmap="gray")
#plt.show()
#print(f"Class: {class_number}, Label: {label}")
#exit;

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#print(f"Using {device} device")

#model = NeuralNetwork().to(device)
#print(model)

#exit()

my_net = Net()
#optimizer = optim.Adam(my_net.parameters(), weight_decay=1e-2)
optimizer = optim.Adam(my_net.parameters(), weight_decay=1e-3)
#optimizer = optim.Adam(my_net.parameters(), weight_decay=5e-5)
#optimizer = optim.SGD(my_net.parameters(), lr=0.01, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

epochs = 10000

for epoch in range(1, epochs + 1):

    size = len(trainloader.dataset)
    number_in_batch = len(trainloader)

    #print(batch_size)
    #print(number_in_batch)
    #print(size)
    #exit()
    print(f"Epoch: {epoch}")

    for i, data in enumerate(trainloader):

        X, y = data

        #print(f"Label: {classes[y[0].item()]}, Class: {y[0].item()}")
        #exit()

        X_output = my_net(X)
        #print(f"Output: {X_output[0]}")
        #print(f"Predicted: {X_output[0].argmax(0)}")

        loss = loss_function(X_output, y)
        #print(f"Loss1: {loss.item()}")

        loss.backward()
        optimizer.step()
        #print(f"Loss2: {loss.item()}")
        optimizer.zero_grad()

        # print(f"X Output Arge max: {X_output.argmax(1)} | y: {y}")
        # print(X_output.argmax(1) == y)
        # print((X_output.argmax(1) == y).type(torch.float).sum().item())
        # exit()
        
        # print statistics
        if not i % print_every or i == number_in_batch - 1:
            loss, current = loss.item(), i * batch_size + len(X)
            correct = (X_output.argmax(1) == y).type(torch.float).sum().item()
            #loss /= number_in_batch
            correct = correct / size
            print(f"Accuracy: {(100*correct)} Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


print('Finished Training')