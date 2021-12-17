# -*- coding: utf-8 -*-
"""
Assignment 2.2 Part (c) test.py 
Author  : Gaurav Jain - 2019CS10349
"""

#Includes
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

#paths
test_path = sys.argv[1]
model_path = sys.argv[2]
pred_path = sys.argv[3]

# DataLoader Class
# if BATCH_SIZE = N, dataloader returns images tensor of size [N, C, H, W] and labels [N]
class ImageDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform=None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [labels,data] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,1:].to_numpy()
            labels = data.iloc[:,0].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
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
        image = np.array(image).astype(np.uint8).reshape((32, 32, 3),order='F')
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
        
        sample = {"images": image, "labels": label}
        return sample

# Data Loader Usage

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
test_tfms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), transforms.Normalize(*stats)])

# Test DataLoader
test_data = test_path # Path to test csv file
test_dataset = ImageDataset(data_csv = test_data, train=False, img_transform=test_tfms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

##########################################################################################################

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            #MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.res1 = Sequential(
            Conv2d(32, 32, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
        )

        self.conv2 = Sequential(
            Conv2d(32, 64, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = Sequential(
            Conv2d(64, 256, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.res2 = Sequential(
            Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
        )

        self.conv4 = Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1,padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2,stride=2),
            MaxPool2d(kernel_size=4,stride=4),
        )

        self.linear1 = Sequential(
            Linear(512 * 1 * 1, 10),
            #ReLU(inplace=True),
        )
        #self.drop = Dropout(p=0.2)
        #self.linear2 = Linear(256,10)
        

    # Defining the forward pass    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)+x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res2(x)+x
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        #x = self.drop(x)
        #x = self.linear2(x)
        
        return x

##########################################################################################################

model = Net()
dic = torch.load(model_path)
model.load_state_dict(dic)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

def predict_model(x_test):
   
    if torch.cuda.is_available():     # converting the data into GPU format
        x_test = x_test.cuda()

    with torch.no_grad():
      output = model(x_test)
    
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    #d = np.sum(predictions==y_test.cpu().detach().numpy())

    return predictions

pred = np.array([])
#acc = 0

for b,sample in enumerate(test_loader):
    images = sample['images']
    #labels = sample['labels']
    
    tmp = predict_model(images)
    pred = np.append(pred,tmp)

    #acc += d

np.savetxt(pred_path,pred)

#print(acc/4000)

"""

#####   #####   #   #   #####   #####   #   #       #####   #####   #####   #   #
#       #   #   #   #   #   #   #   #   #   #         #     #   #     #     ##  #
#  ##   #####   #   #   ####    #####   #   #         #     #####     #     # # #
#   #   #   #   #   #   #  #    #   #    # #        # #     #   #     #     #  ##
#####   #   #   #####   #   #   #   #     #         ###     #   #   #####   #   #

"""