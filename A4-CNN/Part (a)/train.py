#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform

import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
import glob
import os
import sys

#import cv2

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

train_path = sys.argv[1]
test_path = sys.argv[2]

model_path = sys.argv[3]
loss_path = sys.argv[4]
acc_path = sys.argv[5]

# DataLoader Class
# if BATCH_SIZE = N, dataloader returns images tensor of size [N, C, H, W] and labels [N]
class DevanagariDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform = None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [data, labels] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,:-1].to_numpy()
            labels = data.iloc[:,-1].astype(int)
        else:
            images = data.iloc[:,:]
            labels = None
        
        self.images = images
        self.labels = labels
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads image of the given index and performs preprocessing.
        
        INPUT: 
        idx: index of the image to be loaded.
        
        OUTPUT:
        sample: dictionary with keys images (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape(32, 32, 1)
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
#         print(image.shape, label, type(image))
        sample = {"images": image, "labels": label}
        return sample

# Data Loader Usage

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

# Train DataLoader
train_data = train_path # Path to train csv file
train_dataset = DevanagariDataset(data_csv = train_data, train=True, img_transform=img_transforms)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

# Test DataLoader
test_data = test_path # Path to test csv file
test_dataset = DevanagariDataset(data_csv = test_data, train=True, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)


#print('Cuda: {}'.format(torch.cuda.is_available())) #cuda available

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = Sequential(
            Conv2d(32, 64, kernel_size=3, stride=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = Sequential(
            Conv2d(64, 256, kernel_size=3, stride=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=1),
        )

        self.conv4 = Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1),
            ReLU(inplace=True),
        )

        self.linear1 = Sequential(
            Linear(512 * 1 * 1, 256),
            ReLU(inplace=True),
        )
        self.drop = Dropout(p=0.2)
        self.linear2 = Linear(256,46)
        

    # Defining the forward pass    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        
        return x


def train_model(epoch,x_train,y_train):
    model.train()
    tr_loss = 0
    x_train, y_train = Variable(x_train), Variable(y_train)
  
    if torch.cuda.is_available():     # converting the data into GPU format
        x_train = x_train.cuda()
        y_train = y_train.cuda()
    
    optimizer.zero_grad()             # clearing the Gradients of the model parameters
    
    # prediction for training and validation set
    output_train = model(x_train)
    
    # computing the training and validation loss
    loss_train = loss(output_train, y_train)
    
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

    return tr_loss

def predict_model(x_test,y_test):
   
    with torch.no_grad():
      output = model(x_test.cuda())
    
    soft = torch.exp(output).cpu()
    prob = list(soft.numpy())
    predictions = np.argmax(prob, axis=1)
    
    return (np.sum(predictions==y_test.cpu().detach().numpy()))


#torch.manual_seed(51)
model = Net()

optimizer = Adam(model.parameters(), lr=0.0001)
loss = CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()
    loss = loss.cuda()


epochs = 8
loss_file = open(loss_path,'w')
losses = []
acc_file = open(acc_path,'w')
accs = []

for epoch in range(epochs):
    avg_train_loss = 0
    acc = 0
    n = 0

    model.train()
    for batch_idx, sample in enumerate(train_loader):
      images = sample['images']
      labels = sample['labels']
      avg_train_loss += train_model(epoch,images,labels)

    model.eval()
    for b,sample in enumerate(test_loader):
      images = sample['images']
      labels = sample['labels']
      n += len(labels)
      acc += predict_model(images,labels)


    avg_train_loss /= len(train_loader)
    acc /= n
    loss_file.write('{}\n'.format(avg_train_loss))
    losses.append(avg_train_loss)
    acc_file.write('{}\n'.format(acc))
    accs.append(acc)
    #print(epoch+1,avg_train_loss,acc)

loss_file.close()
acc_file.close()

"""
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (15,15)

x = [1,2,3,4,5,6,7,8]
y= losses

plt.xlabel("Number of Epochs")
plt.ylabel("Training Cross-Entropy Loss")
plt.title('Training Loss v/s Epochs (DEVANAGARI)')

plt.plot(x,y,'-mo')
plt.savefig('dplt.png')


y = accs

plt.xlabel("Number of Epochs")
plt.ylabel("Test Accuracy")
plt.title('Test Accuracy v/s Epochs (DEVANAGARI)')


plt.plot(x,y,'-mo')
plt.savefig('dplt1.png')
"""

torch.save(model.state_dict(),model_path)