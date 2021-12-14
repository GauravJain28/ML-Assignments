import numpy as np
import math
import time
import pandas as pd
import sys
import matplotlib.pyplot as plt

train_path  = sys.argv[1]+'train_data_shuffled.csv'
test_path   = sys.argv[1]+'public_test.csv'
output_path = sys.argv[2]

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (15,15)

start = time.time()
#test_labels_path = sys.argv[1]+'toy_dataset_test_labels.csv'

class NeuralNet:
    def __init__(self,seed):
        seed = seed
        np.random.seed(seed)

    class Layer:
        def __init__(self, in_size, out_size):

            #optimizer hyperparameters
            self.gamma = 0.9
            self.v = 0 

            self.nv = 0
            self.ngamma = 0.9

            self.beta = 0.9
            self.eps = 10**(-8)
            self.E = 0

            self.am = 0
            self.av = 0
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.epsa = 10**(-8)

            self.nam = 0
            self.nav = 0
            self.nbeta1 = 0.9
            self.nbeta2 = 0.99
            self.epsna = 10**(-8)
            
            self.in_size = in_size                          # dim(x_l) = m
            self.out_size = out_size                        # dim(y_l) = n
            # Xavier Initializtion of weights for a layer   # dim(w_l) = (m+1)*n 
            self.w = np.float32(np.random.normal(0,1,size=(in_size+1, out_size)) * np.sqrt(2/(in_size + out_size + 1)))  
            self.w = np.float64(self.w)  
            

        def forward(self, input,lr_mode):                                       # forward pass
            input = np.insert(input,0,np.ones(input.shape[0]),axis=1)   # append 1 to input
            self.input = input

            if lr_mode == 2:
                self.w = self.w - self.ngamma*self.nv
            
            return np.dot(input,self.w)

        def backward(self, out_error, learning_rate,lr_mode,iter_num):       # backward pass
            w_temp = np.delete(self.w,0,0)                  # remove first row (bias) from w
            in_error = np.dot(out_error, w_temp.T)
            w_error = np.dot(self.input.T, out_error)

            if lr_mode == 1:                                # momentum
                self.v = self.gamma*self.v+learning_rate*w_error 
                self.w -= self.v   

            elif lr_mode == 2:                              # nesterov
                self.nv = self.ngamma*self.nv+learning_rate*w_error
                self.w -= self.nv 

            elif lr_mode == 3:                              # rmsprop
                self.E = self.beta*self.E + (1-self.beta)*(w_error**2)
                self.w -= learning_rate*w_error/(np.sqrt(self.eps+self.E))

            elif lr_mode == 4:                              # adam
                self.am = self.beta1*self.am + (1-self.beta1)*(w_error)
                am = self.am/(1-self.beta1**iter_num)

                self.av = self.beta2*self.av + (1-self.beta2)*(w_error**2)
                av = self.av/(1-self.beta2**iter_num)

                #print(self.am.shape,self.av.shape)

                self.w -= learning_rate*(am/np.sqrt(self.epsa+av))

            elif lr_mode == 5:                              # nadam
                self.nam = self.nbeta1*self.nam + (1-self.nbeta1)*(w_error)
                nam = self.nam/(1-self.nbeta1**iter_num)

                self.nav = self.nbeta2*self.nav + (1-self.nbeta2)*(w_error**2)
                nav = self.nav/(1-self.nbeta2**iter_num)
                
                grad_upd = self.nbeta1*nam + (1-self.nbeta1)*(w_error)/(1-self.nbeta1**iter_num)

                self.w -= learning_rate*(grad_upd)/(np.sqrt(self.epsna+nav))

            else:
                self.w -= learning_rate*w_error       
            

            return in_error

########################################################################################################

    class ActivationLayer:
        def __init__(self, act_fun, act_prime):
            self.act_fun = act_fun                          #activation function
            self.act_prime = act_prime                      #derivative of activation function
        
        def forward(self, input,lr_mode):
            self.input = input
            #print(self.act_fun(input))
            return self.act_fun(input)                      #activation of linear input
        
        def backward(self, out_error,lr,lr_mode,iter_num):
            return np.multiply(out_error, self.act_prime(self.input))   #derivative w.r.t. activation function 

########################################################################################################

    class SoftmaxLayer:
        def __init__(self, in_size):
            self.in_size = in_size
        
        def forward(self, input,lr_mode):
            self.input = input
            v = np.amax(input,axis=1,keepdims=True)
            #print(input-v)
            tmp = np.exp(input-v)
            tmp = tmp / np.sum(tmp,axis=1,keepdims=True)
            self.output = tmp 
            return self.output
        
        def backward(self, out_error,lr,lr_mode,iter_num):
            
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
    return 1 - (np.tanh(x))**2

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x >= 0).astype('int')

###################################################################################################
#Some Error functions and their derivatives
def mse(pred, y):
    return np.mean(np.power(pred - y, 2))

def mse_prime(pred, y):
    return (pred - y) / y.shape[0]

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
opt_method_dict = {
    0:'Vanilla SGD',
    1:'Momentum',
    2:'Nesterov',
    3:'RMSProp',
    4:'Adam',
    5:'Nadam'
}
act_dict = {
    0:'log-sigmoid',
    1:'tanh',
    2:'ReLU'
}

###################################################################################################

def accuracy(x,y):
    x = np.argmax(x,axis=1)
    y = np.argmax(y,axis=1)
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

def train_model(net,X_train,Y_train,epochs,batchsize,lr,lr_strat,loss_fun,lr_mode=0,details=False,interval=1):
    batchnum = X_train.shape[0]//batchsize
    lr_ = lr
    #lr_mode = 0
    errs = []

    for i in range(epochs):
        err=0
        if lr_strat==1:
            lr_ = lr/np.sqrt(i+1)
        
        for n in range(batchnum):
            mini_x_train = X_train[n*batchsize:(n+1)*batchsize]
            mini_y_train = Y_train[n*batchsize:(n+1)*batchsize]
            iter_num = batchnum*i+n+1
            #print(iter_num)
            output = mini_x_train
            for layer in net:                                                           #forward pass
                output = layer.forward(output,lr_mode)
            #print(output)
            err += loss_fun_dict[loss_fun](output,mini_y_train)                         #error calculation

            out_error = loss_fun_prime_dict[loss_fun](output,mini_y_train)              #derivative of error
            for layer in reversed(net):                                                 #backward pass
                out_error = layer.backward(out_error,lr_,lr_mode,iter_num)
           
        if details:
            output = X_train
            for layer in net:                                                           #forward pass
                output = layer.forward(output,lr_mode)

            error = loss_fun_dict[loss_fun](output,Y_train)                             #error calculation
            errs.append(error)
            # print(np.argmax(output,axis=1))
            if (i+1)%interval==0:
                print(i+1,error,accuracy(output,Y_train))
                
        
    
    return errs
        
###################################################################################################

df = pd.read_csv(train_path,header=None)
test_df = pd.read_csv(test_path,header=None)

Y_df = df[df.columns[-1]].to_numpy()                    #training output data
Y_train = pd.get_dummies(Y_df).to_numpy()               #OHE of training output
df.drop(df.columns[-1],inplace=True,axis=1)

X_train = df.to_numpy()                             #training input data
X_train = X_train/255                               #scaling the data to fit b/w 0 and 1

#labels_df = pd.read_csv(test_labels_path,header=None)
Y_test = pd.get_dummies(test_df[test_df.columns[-1]].to_numpy()).to_numpy()

test_df.drop(test_df.columns[-1],inplace=True,axis=1)
X_test = test_df.to_numpy()
X_test = X_test/255

###################################################################################################

a1 = [512,256,128,64,46]
a2 = [256,46]
params = [
    [10,100,a1,0.1,1,0,1],
    [10,100,a1,0.1,1,1,1],
    [10,100,a1,0.1,1,2,1],
    [10,100,a2,0.1,1,0,1],
    [10,100,a2,0.1,1,1,1],
    [10,100,a2,0.1,1,2,1]
]

def analysis(X_train,Y_train,pp,arch_num,param_num):
    epochs      = pp[0]
    batchsize   = pp[1]
    layers      = pp[2]
    lr_strategy = pp[4]
    lr          = pp[3]
    act_fun     = pp[5]
    loss_fun    = 0
    seed        = pp[6]

    all_errs = []

    #print('Epochs: {}, Batch Size: {}, Layers: {}, lr_stategy: {}, lr: {}, Activation: {}, Loss: {}, Seed: {}'.format(epochs,batchsize,layers,lr_strategy,lr,act_fun,loss_fun,seed))

    for lr_mode in range(6):
        nn = NeuralNet(seed=seed)
        network = []

        network.append(nn.Layer(X_train.shape[1],layers[0]))
        network.append(nn.ActivationLayer(act_fun_dict[act_fun],act_fun_prime_dict[act_fun]))

        for i in range(len(layers)-1):
            network.append(nn.Layer(layers[i],layers[i+1]))
            network.append(nn.ActivationLayer(act_fun_dict[act_fun],act_fun_prime_dict[act_fun]))

        if loss_fun==0:
            network[len(network)-1] = nn.SoftmaxLayer(layers[len(layers)-1])

        #print('Optimizer: {}'.format(opt_method_dict[lr_mode]))
        errs = train_model(network,X_train,Y_train,epochs,batchsize,lr,lr_strategy,loss_fun,lr_mode)
        all_errs.append(errs)

    #plotting
    x = [e for e in range(1,epochs+1)]
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_title("SGD Variants with {} activation - ARCH {}".format(act_dict[act_fun],arch_num))
    ax.plot(x,all_errs[0], label = "Vanilla SGD")
    ax.plot(x,all_errs[1], label = "Momentum")
    ax.plot(x,all_errs[2], "y",label = "Nestorev")
    ax.plot(x,all_errs[3], "g",label = "RMSProp")
    ax.plot(x,all_errs[4], "m",label = "Adam")
    ax.plot(x,all_errs[5], "c",label = "Nadam")
    ax.legend()
    fig.savefig('{}plt_param_{}.png'.format(output_path,param_num))

    #saving tables
    for i in range(6):
        file = open('{}param_{}_method_{}.csv'.format(output_path,param_num,i),'w')
        for j in range(len(all_errs[i])):
            s = '{},{}\n'.format(j+1,all_errs[i][j])
            file.write(s)
        file.close()


###################################################################################################

c = 0
for pp in params:
    c += 1
    if c < 4:
        analysis(X_train,Y_train,pp,1,c)
    else:
        analysis(X_train,Y_train,pp,2,c)
    #break #------------------------#

###################################################################################################

params = [
    [10,[50,100,200,500,1000,2000],a1,0.1,1,1,1],
    [10,[50,100,200,500,1000,2000],a2,0.1,1,1,1]
]

###################################################################################################

def analysis_batch_size(X_train,Y_train,pp,arch_num,param_num):
    epochs      = pp[0]
    batchsizes  = pp[1]
    layers      = pp[2]
    lr_strategy = pp[4]
    lr          = pp[3]
    act_fun     = pp[5]
    loss_fun    = 0
    seed        = pp[6]

    all_errs = []

    #print('Epochs: {}, Batch Sizes: {}, Layers: {}, lr_stategy: {}, lr: {}, Activation: {}, Loss: {}, Seed: {}'.format(epochs,batchsizes,layers,lr_strategy,lr,act_fun,loss_fun,seed))

    for lr_mode in range(6):
        es = []
        for batchsize in batchsizes:
            nn = NeuralNet(seed=seed)
            network = []

            network.append(nn.Layer(X_train.shape[1],layers[0]))
            network.append(nn.ActivationLayer(act_fun_dict[act_fun],act_fun_prime_dict[act_fun]))

            for i in range(len(layers)-1):
                network.append(nn.Layer(layers[i],layers[i+1]))
                network.append(nn.ActivationLayer(act_fun_dict[act_fun],act_fun_prime_dict[act_fun]))

            if loss_fun==0:
                network[len(network)-1] = nn.SoftmaxLayer(layers[len(layers)-1])

            #print('Optimizer: {}'.format(opt_method_dict[lr_mode]))

            errs = train_model(network,X_train,Y_train,epochs,batchsize,lr,lr_strategy,loss_fun,lr_mode)
            es.append(errs[epochs-1])

        all_errs.append(es)

    #plotting
    x = batchsizes
    fig, ax = plt.subplots()
    ax.set_xlabel("Loss after {} Epochs".format(epochs))
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_title("SGD Variants with {} activation - ARCH {}".format(act_dict[act_fun],arch_num))
    ax.plot(x,all_errs[0], label = "Vanilla SGD")
    ax.plot(x,all_errs[1], label = "Momentum")
    ax.plot(x,all_errs[2], "y",label = "Nestorev")
    ax.plot(x,all_errs[3], "g",label = "RMSProp")
    ax.plot(x,all_errs[4], "m",label = "Adam")
    ax.plot(x,all_errs[5], "c",label = "Nadam")
    ax.legend()
    fig.savefig('{}plt_batchsize_param_{}.png'.format(output_path,param_num))

    #saving tables
    for i in range(6):
        file = open('{}batchsize_param_{}_method_{}.csv'.format(output_path,param_num,i),'w')
        for j in range(len(all_errs[i])):
            s = '{},{}\n'.format(batchsizes[j],all_errs[i][j])
            file.write(s)
        file.close()

###################################################################################################

c = 0
for pp in params:
    c += 1
    if c < 2:
        analysis_batch_size(X_train,Y_train,pp,1,c)
    else:
        analysis_batch_size(X_train,Y_train,pp,2,c)
    end = time.time()
    #print('Time: {}'.format(end-start))

###################################################################################################

end = time.time()
#print('Time: {}'.format(end-start))