import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold

np.seterr(divide='ignore', invalid='ignore')

train_path = sys.argv[1]
test_path = sys.argv[2]

def f(pred,Y_train):
    v = np.log(np.sum(Y_train*pred,axis=1))
    #print(np.sum(v))
    return abs(np.sum(v)/Y_train.shape[0])

def read_and_encode(train_path,test_path):
    t = time.time()
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

    print(train)
    #Ensuring consistency of One-Hot Encoding
    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]

    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_train)
    
    X_train = constant_filter.transform(X_train)
    X_test = constant_filter.transform(X_test)

    T = SelectKBest(k=400)
    T.fit(X_train,Y_df.to_numpy())
    X_train = T.transform(X_train)
    X_test = T.transform(X_test)

    #X_train = SelectKBest(score_func=f_regression, k=400).fit_transform(X_train,Y_df)
    Y_train = pd.get_dummies(Y_df).to_numpy()

    #print(time.time()-t)
    return X_train,Y_train,X_test


def mini_batch_grad(X_train,Y_train,X_test,Y_out):
    st = time.time()
    batch_size = 100
    lr0 = 10   
    iter = 100 
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
      
        if (i+1)%50==0:
            h = np.dot(X_train,w)
            pred = softmax(h.T,axis=0).T
            print(i+1,f(pred,Y_train))
        
    h = np.dot(X_test,w)
    pred = softmax(h.T,axis=0).T
    loss = f(pred,Y_out)
    Y_test = np.argmax(pred,axis=1)+1

    end = time.time()
    return Y_test,loss,end-st



st = time.time()
X_train,Y_train,X_test = read_and_encode(train_path,test_path)
X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1)
X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)

#cross validation with new features
k = 10
xFolds = np.array_split(X_train,k)
yFolds = np.array_split(Y_train,k)

sc = 0.0
for i in range(k):
    #train and validation sets
    x_train = xFolds.copy()
    y_train = yFolds.copy()
    x_valid = xFolds[i]
    y_valid = yFolds[i]
    del x_train[i]
    del y_train[i]

    x_train = np.concatenate(x_train,axis=0)
    y_train = np.concatenate(y_train,axis=0)

    y_test,loss,tim = mini_batch_grad(x_train,y_train,x_valid,y_valid)
    y_valid = np.argmax(y_valid,axis=1)+1

    correct = 0
    wrong = 0
    for x in range(y_valid.shape[0]):
        if y_test[x] == y_valid[x]:
            correct += 1
        else:
            wrong += 1

    print(i,correct/y_valid.shape[0],loss)
    sc += correct/y_valid.shape[0]

    #break

end = time.time()

print(sc/k,end-st)
