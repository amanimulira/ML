from torch import nn, optim
import torchvision

model = torchvision.models.resnet18(pretrained=-True)


# Freeze all the parameter in the network

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 10)

# optimize only the classifier

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)