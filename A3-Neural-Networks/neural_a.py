import numpy as np
import math
import time
import pandas as pd
import sys

train_path  = sys.argv[1]+'toy_dataset_train.csv'
test_path   = sys.argv[1]+'toy_dataset_test.csv'
output_path = sys.argv[2]

#test_labels_path = sys.argv[1]+'toy_dataset_test_labels.csv'

class NeuralNet:
    def __init__(self,seed):
        seed = seed
        np.random.seed(seed)

    class Layer:
        def __init__(self, in_size, out_size):
            self.in_size = in_size                          # dim(x_l) = m
            self.out_size = out_size                        # dim(y_l) = n
            # Xavier Initializtion of weights for a layer   # dim(w_l) = (m+1)*n 
            self.w = np.float32(np.random.normal(0,1,size=(in_size+1, out_size)) * np.sqrt(2/(in_size + out_size + 1)))     
            self.w = np.float64(self.w) 
            

        def forward(self, input):                                       # forward pass
            input = np.insert(input,0,np.ones(input.shape[0]),axis=1)   # append 1 to input
            self.input = input
            
            return np.dot(input,self.w)

        def backward(self, out_error, learning_rate):       # backward pass
            w_temp = np.delete(self.w,0,0)                  # remove first row (bias) from w
            in_error = np.dot(out_error, w_temp.T)
            w_error = np.dot(self.input.T, out_error)
            self.w -= learning_rate * w_error               # constant lr strategy

            return in_error

########################################################################################################

    class ActivationLayer:
        def __init__(self, act_fun, act_prime):
            self.act_fun = act_fun                          #activation function
            self.act_prime = act_prime                      #derivative of activation function
        
        def forward(self, input):
            self.input = input
            #print(self.act_fun(input))
            return self.act_fun(input)                      #activation of linear input
        
        def backward(self, out_error,lr):
            return np.multiply(out_error, self.act_prime(self.input))   #derivative w.r.t. activation function 

########################################################################################################

    class SoftmaxLayer:
        def __init__(self, in_size):
            self.in_size = in_size
        
        def forward(self, input):
            self.input = input
            v = np.amax(input,axis=1,keepdims=True)
            #print(input-v)
            tmp = np.exp(input-v)
            tmp = tmp / np.sum(tmp,axis=1,keepdims=True)
            self.output = tmp 
            return self.output
        
        def backward(self, out_error,lr):
            
            return out_error

########################################################################################################
###################################################################################################
# Some activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    z = sigmoid(x)
    return z*(1-z)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x >= 0).astype('int')

###################################################################################################
#Some Error functions and their derivatives
def mse(pred, y):
    return np.mean(np.power(pred - y, 2))

def mse_prime(pred, y):
    return 2 * (pred - y) / pred.shape[0]

def cee(pred,y):
    #pred = np.clip(pred,a_min=10**(-15),a_max=10**15)
    v = np.log(np.sum(pred*y,axis=1,keepdims=True))
    return abs(np.sum(v)/y.shape[0])

def cee_prime(pred,y):
    return (pred-y)/y.shape[0]

###################################################################################################
# mapping from int to loss functions/activation functions 
loss_fun_dict = {
    0:cee, 
    1:mse
}
loss_fun_prime_dict = {
    0:cee_prime, 
    1:mse_prime
}

act_fun_dict = {
    0:sigmoid,
    1:tanh,
    2:relu
}
act_fun_prime_dict = {
    0:sigmoid_prime,
    1:tanh_prime,
    2:relu_prime
}

###################################################################################################

def accuracy(x,y):
    x = np.argmax(x,axis=1)
    #y = np.argmax(y,axis=1)
    c=0
    for i in range(x.shape[0]):
        if x[i]==y[i]:
            c+=1

    return c/x.shape[0]

def pred_model(net,X_test,Y_test=None):
    output = X_test
    for layer in net:                                       #forward pass
        output = layer.forward(output)

    error = cee(output,Y_test)                             #error calculation
    acc = accuracy(output,Y_test)
    print(error,acc)
    return np.argmax(output,axis=1)

###################################################################################################

def train_model(net,X_train,Y_train,epochs,batchsize,lr,lr_strat,loss_fun,details=False,interval=1):
    batchnum = X_train.shape[0]//batchsize
    lr_ = lr

    for i in range(epochs):
        err=0
        if lr_strat==1:
            lr_ = lr/np.sqrt(epochs+1)
        
        for n in range(batchnum):
            mini_x_train = X_train[n*batchsize:(n+1)*batchsize]
            mini_y_train = Y_train[n*batchsize:(n+1)*batchsize]

            output = mini_x_train
            for layer in net:                                                           #forward pass
                output = layer.forward(output)

            err += loss_fun_dict[loss_fun](output,mini_y_train)                         #error calculation

            out_error = loss_fun_prime_dict[loss_fun](output,mini_y_train)              #derivative of error
            for layer in reversed(net):                                                 #backward pass
                out_error = layer.backward(out_error,lr_)
           
 
        if details:
            output = X_train
            for layer in net:                                                           #forward pass
                output = layer.forward(output)

            error = loss_fun_dict[loss_fun](output,Y_train)                             #error calculation

            if (i+1)%interval==0:
                print(i+1,error,accuracy(output,Y_train))
    
    return
        
###################################################################################################

df = pd.read_csv(train_path,header=None)
test_df = pd.read_csv(test_path,header=None)

Y_df = df[df.columns[0]].to_numpy()                 #training output data
Y_train = pd.get_dummies(Y_df).to_numpy()           #OHE of training output
df.drop(df.columns[0],inplace=True,axis=1)
test_df.drop(test_df.columns[0],inplace=True,axis=1)

X_train = df.to_numpy()                             #training input data
X_train = X_train/255                               #scaling the data to fit b/w 0 and 1

X_test = test_df.to_numpy()
X_test = X_test/255

# labels_df = pd.read_csv(test_labels_path,header=None)
# Y_test = pd.get_dummies(labels_df[labels_df.columns[0]].to_numpy()).to_numpy()

###################################################################################################

file = open(sys.argv[3], 'r')
params = file.readlines()
file.close()

epochs      = int(params[0][:-1])
batchsize   = int(params[1][:-1])
layers      = list(map(int,params[2][1:-2].split(',')))
lr_strategy = int(params[3][:-1])
lr          = float(params[4][:-1])
act_fun     = int(params[5][:-1])
loss_fun    = int(params[6][:-1])
seed        = int(params[7])

#print(epochs,batchsize,layers,lr_strategy,lr,act_fun,loss_fun,seed)

nn = NeuralNet(seed=seed)
network = []

network.append(nn.Layer(X_train.shape[1],layers[0]))
network.append(nn.ActivationLayer(act_fun_dict[act_fun],act_fun_prime_dict[act_fun]))

for i in range(len(layers)-1):
    network.append(nn.Layer(layers[i],layers[i+1]))
    network.append(nn.ActivationLayer(act_fun_dict[act_fun],act_fun_prime_dict[act_fun]))

if loss_fun==0:
    network[len(network)-1] = nn.SoftmaxLayer(layers[len(layers)-1])
    

#print(len(network))

###################################################################################################

train_model(network,X_train,Y_train,epochs,batchsize,lr,lr_strategy,loss_fun)   #training the model

for i in range(len(layers)):                                                    #saving weights
    w = network[2*i].w
    np.save('{}w_{}'.format(output_path,i+1),w)

output = X_test                                                                 #final prediction
for layer in network:
    output = layer.forward(output)

Y_out = np.argmax(output,axis=1)
np.save('{}predictions'.format(output_path),Y_out)                              #saving predictions

#print(accuracy(Y_test,Y_out))

###################################################################################################
