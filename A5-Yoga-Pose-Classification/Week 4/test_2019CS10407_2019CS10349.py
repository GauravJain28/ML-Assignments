import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from skimage import io, transform

import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
import glob
import sys
import os
import PIL
from sklearn.model_selection import KFold
import torchvision.models as models

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

device = ("cuda" if torch.cuda.is_available() else "cpu")

modelfile = sys.argv[1]
modelfile += "model.pth"
testfile = sys.argv[2]
subfile = sys.argv[3]

img_test_folder=""


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None, train=True):
        self.df = pd.read_csv(csv_path)
        self.is_train = train
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = {
        "Virabhadrasana":0,
        "Vrikshasana":1,
        "Utkatasana":2,
        "Padahastasana":3,
        "Katichakrasana":4,
        "TriyakTadasana":5,
        "Gorakshasana":6,
        "Tadasana":7,
        "Natarajasana":8,                 
        "Pranamasana":9,
        "ParivrittaTrikonasana":10,
        "Tuladandasana":11,
        "Santolanasana":12,
        "Still":13,
        "Natavarasana":14,
        "Garudasana":15,
        "Naukasana":16,
        "Ardhachakrasana":17,
        "Trikonasana":18,

        }

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df["name"].iloc[index]
        if self.is_train:
            label = self.class2index[self.df["category"].iloc[index]]
        else:
            label = -1
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        sample = {"images": image, "labels": label}
        return sample


BATCH_SIZE = 80
NUM_WORKERS = 20
stats = ((0.4914, 0.4822, 0.5065), (0.2023, 0.1994, 0.2010))

img_test_transforms = transforms.Compose([transforms.Resize(size=(299,299)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(*stats)])

test_data = testfile 
test_dataset = CustomDataset(csv_path = test_data, images_folder = img_test_folder, transform=img_test_transforms, train=False)

class Net_drop_1(Module):   
    def __init__(self):
        super(Net_drop_1, self).__init__()

        self.cnn_layers = Sequential(
            
            Conv2d(3, 32, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            Dropout(p = 0.2),
            
            Conv2d(32, 64, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p = 0.2),
            
            Conv2d(64, 128, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p = 0.2),
            
            Conv2d(128, 128, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p = 0.2),
            
            Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p = 0.2),
            
            Conv2d(256, 512, kernel_size=3, stride=1,padding=1),
            ReLU(inplace=True),
            Dropout(p = 0.2),
        )

        self.linear_layers = Sequential(
            Linear(512*4*4 , 512),
            ReLU(inplace=True),
            Dropout(p = 0.2),
            Linear(512, 64),
            ReLU(inplace=True),
            Dropout(p = 0.2),
            Linear(64 , 19),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    
def predict_n(x, model):    
    model.eval()
    x_train= Variable(x)
    
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        
    output_train = model(x_train)
    output_train = torch.argmax(output_train, dim = 1)
    return output_train




class Inception_Model(Module):
    def __init__(self, pretrained=True):
        super(Inception_Model,self).__init__()
        
        self.m = models.inception_v3(pretrained=True)
        self.m.fc = nn.Linear(self.m.fc.in_features, 19)

    def forward(self, xb):
        return self.m(xb)
    
cnnmodel = Inception_Model()   
dic = torch.load(modelfile)
cnnmodel.load_state_dict(dic)
cnnmodel.eval()


print(sum(p.numel() for p in cnnmodel.parameters()))

if torch.cuda.is_available():
    cnnmodel = cnnmodel.cuda()

test_loader = torch.utils.data.DataLoader(
              test_dataset,
              batch_size=BATCH_SIZE,  num_workers = NUM_WORKERS, shuffle = False)

predictions = torch.Tensor([])

if torch.cuda.is_available():
    predictions = predictions.cuda()
    
for batch_idx, sample in enumerate(test_loader):
    images = sample['images']
    
    temp = predict_n(images, cnnmodel)
    predictions = torch.cat((predictions,temp),0)

predictions=predictions.cpu().detach().numpy()

classif = {
        "Virabhadrasana":0,
        "Vrikshasana":1,
        "Utkatasana":2,
        "Padahastasana":3,
        "Katichakrasana":4,
        "TriyakTadasana":5,
        "Gorakshasana":6,
        "Tadasana":7,
        "Natarajasana":8,                 
        "Pranamasana":9,
        "ParivrittaTrikonasana":10,
        "Tuladandasana":11,
        "Santolanasana":12,
        "Still":13,
        "Natavarasana":14,
        "Garudasana":15,
        "Naukasana":16,
        "Ardhachakrasana":17,
        "Trikonasana":18,

        }
    
inv_map = {v: k for k, v in classif.items()}

pred = [inv_map[letter] for letter in predictions]
df1 = pd.read_csv("test.csv")
df1["category"] = pred
df1.drop(df1.tail(1).index,inplace=True)
df1.to_csv(path_or_buf=subfile, columns=["name", "category"],index=False)

