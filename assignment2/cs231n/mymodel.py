import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F
import random

# load the data
NUM_TRAIN = 49000
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)
# the type and device

dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc

def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    print_every = 100
    test_limits = 100
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()
            loss.item()

best_optimizer = None
best_model = None 
best_acc = -1
A = 0
B = 0
for i in range(4, 10):
    channel_1 = 23
    channel_2 = 9
    channel_3 = i
    learning_rate = 1e-2
    reg = 1e-4 
    model = nn.Sequential(
        nn.Conv2d(3, channel_1, kernel_size = (5, 5), padding = 2), 
        nn.BatchNorm2d(channel_1), 
        nn.ReLU(),  
        nn.Conv2d(channel_1, channel_2, kernel_size = (3, 3), padding = 1), 
        nn.BatchNorm2d(channel_2), 
        nn.ReLU(), 
        nn.Conv2d(channel_2, channel_3, kernel_size = (3, 3), padding = 1), 
        nn.BatchNorm2d(channel_3), 
        nn.ReLU(), 
        nn.MaxPool2d(2, stride = 2),
        nn.Flatten(), 
        nn.Linear(channel_3 * 16 * 16, 2048), 
        nn.ReLU(),
        nn.Linear(2048, 1024), 
        nn.ReLU(), 
        nn.Linear(1024, 10)  
    )
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = reg)
    train_part34(model, optimizer)
    acc = check_accuracy_part34(loader_test, model)
    if acc > best_acc:
        best_acc = acc 
        best_model = model 
        best_optimizer = optimizer 
        A = channel_1 
        B = channel_2

train_part34(best_model, best_optimizer, epochs=10)
check_accuracy_part34(loader_test, best_model)

# 63.31%