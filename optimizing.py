import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda



training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)

test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

"""
epochs = the number of times to iterate over the dataset 

batch size = the number of data samples propagated through the network before the parameter are updated

learning rate how much to update models parameters at each batch/epoch.

"""

# hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5


# optimization loop

# the optimization loop has 2 parts:

    # the train loop : iterate over the training dataset and try to converge to optimal parameters.
    # the validation/ test loop : iterate over the test dataset to check if model performace is improving.

# this is the loss function
loss_fn = nn.CrossEntropyLoss()

# optimization is basically adjusting the models parameters to reduce the model error in each training step.

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#inside the training loop

"""Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; 
to prevent double-counting, we explicitly zero them at each iteration.
Backpropagate the prediction loss with a call to loss.backwards(). PyTorch deposits the gradients 
of the loss w.r.t. each parameter.
Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients 
collected in the backward pass"""

def train_loop(dataloader, model, loss_fn, optimizier):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizier.zero_grad()
        loss.backward()
        optimizier.step()


        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0


    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()


        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")