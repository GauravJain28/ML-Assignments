import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax

train_path = sys.argv[1]
test_path = sys.argv[2]

def f(pred,Y_train):
    v = np.log(np.sum(Y_train*pred,axis=1))
    #print(np.sum(v))
    return abs(np.sum(v)/Y_train.shape[0])

def read_and_encode(train_path,test_path):
    train = pd.read_csv(train_path, index_col = 0)    
    test = pd.read_csv(test_path, index_col = 0)
        
    Y_df = train['Length of Stay']

    train = train.drop(columns = ['Length of Stay'])
    #Ensuring consistency of One-Hot Encoding
    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]

    Y_train = pd.get_dummies(Y_df).to_numpy()
    
    return X_train,Y_train,X_test

#------------------------------------------------------------------------------------------#
def batch_grad(k,iter,lr0,alpha,beta):
    st = time.time()
    X_train,Y_train,X_test = read_and_encode(train_path,test_path)
    X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1)
    X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)

    # M1: Constant learning rate 
    if k==1:
        w = np.zeros((X_train.shape[1],8))

        for i in range(iter):
            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0)
            pred = pred.T
            grad = np.dot(X_train.T,pred-Y_train)/(X_train.shape[0])
            w  = w - lr0*grad

        h = np.dot(X_train,w)
        pred = softmax(h.T,axis=0)
        pred = pred.T

        loss = f(pred,Y_train)

    # M2: Adaptive learning rate - alpha/sqrt(iter_no)
    elif k==2:
        w = np.zeros((X_train.shape[1],8))

        for i in range(iter):
            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0).T
            grad = np.dot(X_train.T,pred-Y_train)/(X_train.shape[0])
            w  = w - (lr0*grad)/np.sqrt(i+1)
            
        h = np.dot(X_train,w)
        pred = softmax(h.T,axis=0)
        pred = pred.T

        loss = f(pred,Y_train)

    # M3: alpha-beta backtracking line search
    elif k==3:
        #print('alpha-beta Backtracking line search')
        w = np.zeros((X_train.shape[1],8))
        for i in range(iter):

            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0)
            pred = pred.T
            grad = np.dot(X_train.T,pred-Y_train)/(X_train.shape[0])

            lr = lr0
            h1 = np.dot(X_train,w-lr*grad)
            pred1 = softmax(h1.T,axis=0)
            pred1 = pred1.T
            
            while f(pred1,Y_train) > f(pred,Y_train) - lr*alpha*np.square(np.linalg.norm(grad)):
                lr = lr*beta
                h1 = np.dot(X_train,w-lr*grad)
                pred1 = softmax(h1.T,axis=0)
                pred1 = pred1.T
            
            #print(i,lr)
            w  = w - (lr*grad)

        h = np.dot(X_train,w)
        pred = softmax(h.T,axis=0)
        pred = pred.T

        loss = f(pred,Y_train)

    end = time.time()
    print(loss,str(end-st))

#------------------------------------------------------------------------------------------#
def mini_batch_grad(k,iter,lr0,batch_size,alpha,beta):
    st = time.time()
    X_train,Y_train,X_test = read_and_encode(train_path,test_path)
    X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1)
    X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)

    # M1: Constant learning rate 
    if k==1:
        batch_no = X_train.shape[0]//batch_size 
        w = np.zeros((X_train.shape[1],8))

        for i in range(iter):
            
            for n in range(batch_no):
                mini_X_train = X_train[n*batch_size:(n+1)*batch_size]
                mini_Y_train = Y_train[n*batch_size:(n+1)*batch_size]
                #print(mini_X_train.shape,mini_Y_train.shape)
                h = np.dot(mini_X_train,w)
                pred = softmax(h.T,axis=0)
                pred = pred.T
                grad = np.dot(mini_X_train.T,pred-mini_Y_train)/(mini_X_train.shape[0])
                w  = w - lr0*grad


        h = np.dot(X_train,w)
        pred = softmax(h.T,axis=0)
        pred = pred.T

        loss = f(pred,Y_train)

    # M2: Adaptive learning rate - alpha/sqrt(iter_no)
    elif k==2:
        
        batch_no = X_train.shape[0]//batch_size 
        w = np.zeros((X_train.shape[1],8))

        for i in range(iter):
            
            for n in range(batch_no):
                mini_X_train = X_train[n*batch_size:(n+1)*batch_size]
                mini_Y_train = Y_train[n*batch_size:(n+1)*batch_size]
                #print(mini_X_train.shape,mini_Y_train.shape)
                h = np.dot(mini_X_train,w)
                pred = softmax(h.T,axis=0)
                pred = pred.T
                grad = np.dot(mini_X_train.T,pred-mini_Y_train)/(mini_X_train.shape[0])
                w  = w - (lr0*grad)/np.sqrt(i+1)
            

        h = np.dot(X_train,w)
        pred = softmax(h.T,axis=0)
        pred = pred.T

        loss = f(pred,Y_train)

    # M3: alpha-beta backtracking line search
    elif k==3:
        
        batch_no = X_train.shape[0]//batch_size 
        w = np.zeros((X_train.shape[1],8))
        #print(lr0,alpha,beta)

        for i in range(iter):

            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0)
            pred = pred.T
            grad = np.dot(X_train.T,pred-Y_train)/(X_train.shape[0])

            lr = lr0
            h1 = np.dot(X_train,w-lr*grad)
            pred1 = softmax(h1.T,axis=0)
            pred1 = pred1.T
            
            while f(pred1,Y_train) > f(pred,Y_train) - lr*alpha*np.square(np.linalg.norm(grad)):
                lr = lr*beta
                h1 = np.dot(X_train,w-lr*grad)
                pred1 = softmax(h1.T,axis=0)
                pred1 = pred1.T
            
            #print(i,lr)

            for n in range(batch_no):
                mini_X_train = X_train[n*batch_size:(n+1)*batch_size]
                mini_Y_train = Y_train[n*batch_size:(n+1)*batch_size]
                h = np.dot(mini_X_train,w)
                pred = softmax(h.T,axis=0)
                pred = pred.T
                grad = np.dot(mini_X_train.T,pred-mini_Y_train)/(mini_X_train.shape[0])
                w  = w - (lr*grad)
            

        h = np.dot(X_train,w)
        pred = softmax(h.T,axis=0)
        pred = pred.T

        loss = f(pred,Y_train)

    end = time.time()
    print(loss,str(end-st))

#hyperparameter search
#--------------------------------------------------------------------#

#batch gradient descent v/s min-batch gradient descent
# batch_grad(1,200,0.03,0,0)
# mini_batch_grad(1,200,0.03,200,0,0)

#--------------------------------------------------------------------#

#search for good batch_size in mini-batch gradient descent 
#Loss-runtime tradeoff is observed. Batch_size=300 is a good value

# print('Batch_Size search')
# bsz = [50,100,200,300,500,800,1000,2000]
# for size in bsz:
#     mini_batch_grad(1,200,0.03,size,0,0)

#--------------------------------------------------------------------#

#comparison between three strategies-
#1. fixed learning rate
#2. adaptive learning rate
#3. alpha-beta backtracking search

# mini_batch_grad(1,50,2.5,300,0,0)
# mini_batch_grad(2,50,2.5,300,0,0)
# mini_batch_grad(3,50,2.5,300,0.4,0.9)

#--------------------------------------------------------------------#

#search for initial learning rate for adaptive learning rate  
# ilrs = [2,4,6,8,10,12,15]
# for lr in ilrs:
#     mini_batch_grad(2,100,lr,100,0,0)   

#--------------------------------------------------------------------#

#search for app no of iterations 
#itrs = [100,200,400,800,1200]
#for it in itrs:
#    mini_batch_grad(2,it,12,100,0,0)

#final tuned parameters
mini_batch_grad(2,600,12,100,0.4,0.75)