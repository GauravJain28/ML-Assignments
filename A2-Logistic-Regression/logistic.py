import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import VarianceThreshold

np.seterr(divide='ignore', invalid='ignore')

st = time.time()
mode = sys.argv[1]
train_path = sys.argv[2]
test_path = sys.argv[3]

def f(pred,Y_train):
    pred = np.clip(pred,a_min=10**(-15),a_max=10**15)
    v = np.log(np.sum(Y_train*pred,axis=1))
    #v = np.clip(v,a_min=10**(-50),a_max=10**(50))
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

def read_and_encode_d(train_path,test_path):
    #t = time.time()
    train = pd.read_csv(train_path, index_col = 0)    
    test = pd.read_csv(test_path, index_col = 0)
        
    Y_df = train['Length of Stay']

    train.drop('Birth Weight',axis=1,inplace=True)
    test.drop('Birth Weight',axis=1,inplace=True)

    train['n1'] = train['Facility Name'] + 1000*train['APR DRG Code']
    test['n1'] = test['Facility Name'] + 1000*test['APR DRG Code']

    train['n2'] = train['Facility Name'] + 1000*train['CCS Procedure Code']
    test['n2'] = test['Facility Name'] + 1000*test['CCS Procedure Code']

    # train['n3'] = train['Facility Name'] + 1000*train['CCS Diagnosis Code']
    # test['n3'] = test['Facility Name'] + 1000*test['CCS Diagnosis Code']

    # train['n4'] = train['Zip Code - 3 digits'] + 1000*train['APR DRG Code']
    # test['n4'] = test['Zip Code - 3 digits'] + 1000*test['APR DRG Code']


    #ss = ['CCS Procedure Code','APR DRG Code','CCS Diagnosis Code','APR MDC Code','APR Severity of Illness Code','APR Risk of Mortality','APR Medical Surgical Description']
    ss = ['n1','n2']
    for s in ss:
      for ll in pd.unique(train[s]):
          #print(ll)
          m = train.loc[train[s]==ll]['Length of Stay'].median()
          train[s] = train[s].mask(train[s]==ll,m)
          test[s] = test[s].mask(test[s]==ll,m)

      k = train[s]
      l = test[s]
      train.drop(s,axis=1,inplace=True)
      test.drop(s,axis=1,inplace=True)

      train.insert(0,s,k)
      test.insert(0,s,l)

    train = train.drop(columns = ['Length of Stay'])

    #print(train)
    #Ensuring consistency of One-Hot Encoding
    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]

    # constant_filter = VarianceThreshold(threshold=0)
    # constant_filter.fit(X_train)
    
    # X_train = constant_filter.transform(X_train)
    # X_test = constant_filter.transform(X_test)
    
    # #print('selection of 400 features')
    # T = SelectKBest(k=400)    
    # T.fit(X_train,Y_df.to_numpy())
    # X_train = T.transform(X_train)
    # X_test = T.transform(X_test)

    Y_train = pd.get_dummies(Y_df).to_numpy()

    #print(time.time()-t)
    return X_train,Y_train,X_test

#------------------------------------------------------------------------------------------------------#    
if mode == 'a':
    X_train,Y_train,X_test = read_and_encode(train_path,test_path)
    X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1)
    X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)

    file = open(sys.argv[4],'r')
    par = file.readlines()
    file.close()

    k = int(par[0])

    # M1: Constant learning rate 
    if k==1:
        iter = int(par[2])
        alpha = float(par[1])

        w = np.zeros((X_train.shape[1],8))

        for i in range(iter):
            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0)
            pred = pred.T
            grad = np.dot(X_train.T,pred-Y_train)/(X_train.shape[0])
            w  = w - alpha*grad

        np.savetxt(sys.argv[6],w.flatten())
        h = np.dot(X_test,w)
        pred = softmax(h.T,axis=0).T
        Y_test = np.argmax(pred,axis=1)+1
        np.savetxt(sys.argv[5],Y_test)

    # M2: Adaptive learning rate - alpha/sqrt(iter_no)
    elif k==2:
        iter = int(par[2])
        alpha = float(par[1])
        w = np.zeros((X_train.shape[1],8))

        for i in range(iter):
            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0).T
            grad = np.dot(X_train.T,pred-Y_train)/(X_train.shape[0])
            w  = w - (alpha*grad)/np.sqrt(i+1)
            

        np.savetxt(sys.argv[6],w.flatten())
        h = np.dot(X_test,w)
        pred = softmax(h.T,axis=0).T
        Y_test = np.argmax(pred,axis=1)+1
        np.savetxt(sys.argv[5],Y_test)

    # M3: alpha-beta backtracking line search
    elif k==3:
        #print('alpha-beta Backtracking line search')
        iter = int(par[2])
        pp = par[1].split(',')
        lr0 = float(pp[0])
        alpha = float(pp[1])
        beta = float(pp[2])

        
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
           
            w  = w - (lr*grad)


        np.savetxt(sys.argv[6],w.flatten())
        h = np.dot(X_test,w)
        pred = softmax(h.T,axis=0).T
        Y_test = np.argmax(pred,axis=1)+1
        np.savetxt(sys.argv[5],Y_test)

#--------------------------------------------------------------------------------------------------#
elif mode == 'b':
    X_train,Y_train,X_test = read_and_encode(train_path,test_path)
    X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1)
    X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)

    file = open(sys.argv[4],'r')
    par = file.readlines()
    file.close()

    k = int(par[0])

    # M1: Constant learning rate 
    if k==1:
        iter = int(par[2])
        alpha = float(par[1])
        batch_size = int(par[3])
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
                w  = w - alpha*grad


        np.savetxt(sys.argv[6],w.flatten())
        h = np.dot(X_test,w)
        pred = softmax(h.T,axis=0).T
        Y_test = np.argmax(pred,axis=1)+1
        np.savetxt(sys.argv[5],Y_test)

    # M2: Adaptive learning rate - alpha/sqrt(iter_no)
    elif k==2:
        iter = int(par[2])
        alpha = float(par[1])
        batch_size = int(par[3])
        batch_no = X_train.shape[0]//batch_size 
        w = np.zeros((X_train.shape[1],8))

        for i in range(iter):
            
            for n in range(batch_no):
                mini_X_train = X_train[n*batch_size:(n+1)*batch_size]
                mini_Y_train = Y_train[n*batch_size:(n+1)*batch_size]
                h = np.dot(mini_X_train,w)
                pred = softmax(h.T,axis=0)
                pred = pred.T
                grad = np.dot(mini_X_train.T,pred-mini_Y_train)/(mini_X_train.shape[0])
                w  = w - (alpha*grad)/np.sqrt(i+1)
            
        np.savetxt(sys.argv[6],w.flatten())
        h = np.dot(X_test,w)
        pred = softmax(h.T,axis=0).T
        Y_test = np.argmax(pred,axis=1)+1
        np.savetxt(sys.argv[5],Y_test)

    # M3: alpha-beta backtracking line search
    elif k==3:
        iter = int(par[2])
        pp = par[1].split(',')
        lr0 = float(pp[0])
        alpha = float(pp[1])
        beta = float(pp[2])
        batch_size = int(par[3])
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
            
            

        np.savetxt(sys.argv[6],w.flatten())

        h = np.dot(X_test,w)
        pred = softmax(h.T,axis=0).T
        #pred = np.clip(pred,a_min=10**(-15),a_max=10**15)
        Y_test = np.argmax(pred,axis=1)+1
        np.savetxt(sys.argv[5],Y_test)

#-------------------------------------------------------------------------------------------------#
elif mode=='c':
    X_train,Y_train,X_test = read_and_encode(train_path,test_path)
    X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1)
    X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)    

    iter = 800
    lr0 = 10
    batch_size = 100
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
            
            #print(i+1,alpha/np.sqrt(i+1))
        if (i+1)%50==0:
            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0).T
            #print(i+1,f(pred,Y_train))

            np.savetxt(sys.argv[5],w.flatten())
            h = np.dot(X_test,w)
            pred = softmax(h.T,axis=0).T
            Y_test = np.argmax(pred,axis=1)+1
            np.savetxt(sys.argv[4],Y_test)

        end = time.time()
        if end-st > 540.0:
            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0).T
            #print(i+1,f(pred,Y_train))

            np.savetxt(sys.argv[5],w.flatten())
            h = np.dot(X_test,w)
            pred = softmax(h.T,axis=0).T
            Y_test = np.argmax(pred,axis=1)+1
            np.savetxt(sys.argv[4],Y_test)
            break

    # h = np.dot(X_train,w)
    # pred = softmax(h.T,axis=0).T
    # print(i+1,f(pred,Y_train))

    # np.savetxt(sys.argv[5],w.flatten())
    # h = np.dot(X_test,w)
    # pred = softmax(h.T,axis=0).T
    # Y_test = np.argmax(pred,axis=1)+1
    # np.savetxt(sys.argv[4],Y_test)

#-------------------------------------------------------------------------------------------------# 
elif mode=='d':
    X_train,Y_train,X_test = read_and_encode_d(train_path,test_path)
    X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1)
    X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)    

    iter = 600
    lr0 = 10
    #alpha = 0.4
    #beta = 0.75
    batch_size = 100
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
            
            #print(i+1,alpha/np.sqrt(i+1))
        if (i+1)%50==0:
            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0).T
            #print(i+1,f(pred,Y_train))

            np.savetxt(sys.argv[5],w.flatten())
            h = np.dot(X_test,w)
            pred = softmax(h.T,axis=0).T
            Y_test = np.argmax(pred,axis=1)+1
            np.savetxt(sys.argv[4],Y_test)

        end = time.time()
        if end-st > 840.0:
            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0).T
            #print(i+1,f(pred,Y_train))

            np.savetxt(sys.argv[5],w.flatten())
            h = np.dot(X_test,w)
            pred = softmax(h.T,axis=0).T
            Y_test = np.argmax(pred,axis=1)+1
            np.savetxt(sys.argv[4],Y_test)
            break

else:
    print('Invalid option')
    
end = time.time()
#print(str(end-st)+' s')
