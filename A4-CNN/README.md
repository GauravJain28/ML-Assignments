# Assignment 4 : Convolutional Neural Networks

The problem statement of the assignment is present [here](A4-PS.pdf).

In this assignment, we work with Convolutional Neural Networks (CNN) and deep learning using the deep learning framework PyTorch. We perform image classification on two datasets- Devanagari handwritten characters dataset and CIFAR10 dataset.\
The Devanagari dataset is present in this [link](https://drive.google.com/drive/folders/1WbcZps9Khe6vjNjlj7fEg6r2J1lHXYsn?usp=sharing).\
The CIFAR10 dataset is present in this [link](https://drive.google.com/drive/folders/1PFOTaRsoIH-GePsX4787y1qpcT_eYBkh?usp=sharing).

The assignment consists of three parts, in which we perform image classification task:
- A simple CNN architecture is implemented and performance is evaluated on Devanagari dataset.
- A slightly more complex architecture is evaluated on CIFAR10 dataset. 
- Competitive part on CIFAR-10 dataset. 

All the parts have to be implemented in
PyTorch.
The following modules are used- Python3.7, glob, os, collections, PyTorch, torchvision, numpy, pandas, skimage, scipy, scikit-learn, matplotlib.

<!-- ### Part (a) CNN Architecture:
```
i. CONV1 (2D Convolution Layer) in_channels = 1, out_channels = 32, kernel = 3×3, stride = 1.
ii. BN1 2D Batch Normalization Layer
iii. RELU ReLU Non-Linearity
iv. POOL1 (MaxPool Layer) kernel size=2×2, stride=2.
v.  CONV2 (2D Convolution Layer) in channels = 32, out channels = 64, kernel=3×3, stride = 1.
vi. BN1 2D Batch Normalization Layer
vii. RELU ReLU Non-Linearity
viii. POOL2 (MaxPool Layer) kernel size=2×2, stride=2.
ix. CONV3 (2D Convolution Layer) in channels = 64, out channels = 256, kernel=3×3, stride = 1.
x. BN1 2D Batch Normalization Layer
xi. RELU ReLU Non-Linearity
xii. POOL3 (MaxPool Layer) kernel size=2×2, stride=1.
xiii. CONV4 (2D Convolution Layer) in channels = 256, out channels = 512, kernel=3×3, stride =
1.
xiv. RELU ReLU Non-Linearity
xv. FC1 (Fully Connected Layer) output = 256
xvi. RELU ReLU Non-Linearity
xvii. DROPOUT Dropout layer with p = 0:2
xviii. FC2 (Fully Connected Layer) output = 46
``` -->



## How to run the code?

- Model Training:
    ```
    python3 train.py <path_train_data.csv> <name_trained_model.pth> <loss.txt> <accuracy.txt>
    ```
- Prediction:
    ```
    python3 test.py <path_test_data.csv> <path_trained_model.pth> <pred.txt>
    ```

A  brief report on the different parts of the assignment is present [here](A4-Report.pdf).

## Author
* Gaurav Jain [[GitHub](https://github.com/GauravJain28/)]