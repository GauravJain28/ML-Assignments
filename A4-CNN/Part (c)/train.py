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
import time

#import cv2

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Dropout2d
from torch.optim import Adam, SGD

train_path = sys.argv[1]
test_path = sys.argv[2]
model_path = sys.argv[3]
loss_path = sys.argv[4]
acc_path = sys.argv[5]


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
            images = data.iloc[:,:]
            labels = None
        
        self.images = images
        self.labels = labels
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """nimages (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape((32, 32, 3),order="F")
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        ##print(image)
        image = self.img_transform(image)
        
        sample = {"images": image, "labels": label}
        return sample

# Data Loader Usage

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_tfms = transforms.Compose([transforms.ToPILImage(),
                                 transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])
test_tfms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), transforms.Normalize(*stats)])

# Train DataLoader
train_data = train_path # Path to train csv file
# train_data = "/mnt/scratch1/siy197580/COL341/cifar/train_data.csv"
# test_data = "/mnt/scratch1/siy197580/COL341/cifar/public_test.csv"
train_dataset = ImageDataset(data_csv = train_data, train=True, img_transform=train_tfms)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

# Test DataLoader
test_data = test_path # Path to test csv file
# train_data = "/mnt/scratch1/siy197580/COL341/cifar/train_data.csv"
# test_data = "/mnt/scratch1/siy197580/COL341/cifar/public_test.csv"
test_dataset = ImageDataset(data_csv = test_data, train=True, img_transform=test_tfms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)


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


def train_model(x_train,y_train):
    #model.train()
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
    #print(loss_train)
  
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

    return tr_loss

def predict_model(x_test,y_test):
   
    if torch.cuda.is_available():     # converting the data into GPU format
        x_test = x_test.cuda()

    with torch.no_grad():
      output = model(x_test)
    
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    
    return (np.sum(predictions==y_test.cpu().detach().numpy()))

#torch.manual_seed(51)
model = Net()

#optimizer = SGD(model.parameters(),momentum=0.9,lr=0.1,nesterov=True)
#Adam(model.parameters(),lr=1e-3)
#SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=4e-5,nesterov=True)
loss = CrossEntropyLoss()
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,nesterov=True)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

if torch.cuda.is_available():
    model = model.cuda()
    loss = loss.cuda()


st = time.time()
epochs = 25
loss_file = open(loss_path,'w')
acc_file = open(acc_path,'w')

losses = []
accs = []


for epoch in range(epochs):
    avg_train_loss = 0
    acc = 0
    n = 0
    
    model.train()
    for batch_idx, sample in enumerate(train_loader):
      images = sample['images']
      labels = sample['labels']
      avg_train_loss += train_model(images,labels)
      scheduler.step()

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

    print(epoch+1,avg_train_loss,acc,time.time())
    if time.time()-st > 1600:
        break

loss_file.close()
acc_file.close()

"""
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (15,15)

x = [i+1 for i in range(epochs)]
y= losses

plt.xlabel("Number of Epochs")
plt.ylabel("Training Cross-Entropy Loss")
plt.title('Training Loss v/s Epochs (CIFAR10)')

plt.plot(x,y,'-mo')
plt.savefig('c10plt.png')

y = accs

plt.xlabel("Number of Epochs")
plt.ylabel("Test Accuracy")
plt.title('Test Accuracy v/s Epochs (CIFAR10)')


plt.plot(x,y,'-mo')
plt.savefig('c10plt1.png')


print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
"""


torch.save(model.state_dict(),model_path)


"""
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(model)

pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_params

"""

