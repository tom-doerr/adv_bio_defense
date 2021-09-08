#!/usr/bin/env python3

'''
This short and simple script trains a Pytorch model on Imagenette.
'''

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchsummary import summary

import argparse
from pathlib import Path

import torch.nn.functional as F

def add_delta(x, layer_id, delta_dict, use_delta=True):
    if not use_delta:
        return x
    if layer_id in delta_dict:
        delta = delta_dict[layer_id]
        delta[0,0,0]
        return x + delta

    x_shape = x[0].shape
    delta = torch.rand(x_shape) * 1e-3
    delta = delta.to(x.device)
    delta_dict[layer_id] = delta
    return add_delta(x, layer_id, delta_dict)

# Network suitable for Imagnet that can handle images of different sizes..
class Net(nn.Module):
    '''
    This class implements a Convolutional Neural Network on Imagenette.
    The network takes input images of resolution 224 x 224.
    '''

    def __init__(self, use_delta):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.delta_dict = {}
        self.use_delta = use_delta

    def forward(self, x):
        x = self.conv1(x)
        x = add_delta(x, 0, self.delta_dict, use_delta=self.use_delta)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = add_delta(x, 1, self.delta_dict, use_delta=self.use_delta)
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


parser = argparse.ArgumentParser(description='Train a Pytorch model on Imagenette')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for loading data')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for SGD')
parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
parser.add_argument('--data_root', type=str, default='imagenette2', help='Path to data')
parser.add_argument('--model_path', type=str, default='models/model.pt', help='Path to save the model')
parser.add_argument('--use_delta', action='store_true', default=False)

args = parser.parse_args()

# Set up data
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Add data augmentation to the data.
data_transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_data = datasets.ImageFolder(os.path.join(args.data_root, 'train'), transform=data_transform_train)
test_data = datasets.ImageFolder(os.path.join(args.data_root, 'val'), transform=data_transform)

train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# Set up model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Net(args.use_delta)
model.to(device)

# Set up optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Set up logging
log_interval = args.log_interval

if __name__ == '__main__':
    # Train
    for epoch in range(args.epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                print('Epoch: {}/{} - Batch: {}/{} - Loss: {:.6}'.format(
                    epoch + 1, args.epochs, i + 1, len(train_data_loader), loss.item()))

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(test_data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the {} test images: {} %'.format(len(test_data_loader) * args.batch_size,
                                                                               100 * correct / total))

    # Save model
    torch.save(model.state_dict(), args.model_path)
    if args.use_delta:
        torch.save(model, args.model_path.replace('model.pt', 'model_delta_full.pt'))
    else:
        torch.save(model, args.model_path.replace('model.pt', 'model_full.pt'))
