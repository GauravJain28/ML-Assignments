import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


if sys.argv[1] == 'a':

    '''
    data processing part
    '''

    df = pd.read_csv(sys.argv[2])
    df.drop(df.columns[0],axis=1,inplace=True)
    #print(df.head())
    n = df.shape[1]-1
    y_df = df[df.columns[n]]
    df.drop(df.columns[n],axis=1,inplace=True)
    X_input = df.to_numpy()
    Y_input = y_df.to_numpy(dtype='float64')
    Y_input = Y_input.reshape(Y_input.shape[0],1)
    # t = np.dot(Y_input.T,Y_input)
    #print(X_input.shape)
    # print(Y_input)

    '''
    Calculating w using moore-penrose pseudo inverse formula
        w = Inv(X.T X) X.T Y
    Inv calculated using np.linalg.inv function
    '''

    X_input = np.insert(X_input,0,np.ones(X_input.shape[0]),axis=1)
    inv = np.dot(X_input.T,X_input)
    h = np.dot(np.linalg.inv(inv),X_input.T)
    w = np.dot(h,Y_input)
    np.savetxt(sys.argv[5],w)

    '''
    Processing test data
    '''

    df = pd.read_csv(sys.argv[3])
    df.drop(df.columns[0],axis=1,inplace=True)
    #print(df.head())
    X_test = df.to_numpy()
    X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)

    '''
    Computing the predictions using w
        Y = X w
    '''

    Y_test = np.dot(X_test,w)
    np.savetxt(sys.argv[4],Y_test)

elif sys.argv[1] == 'b':

    '''
    data processing part
    '''

    df = pd.read_csv(sys.argv[2])
    df.drop(df.columns[0],axis=1,inplace=True)
    #print(df.head())
    n = df.shape[1]-1
    y_df = df[df.columns[n]]
    df.drop(df.columns[n],axis=1,inplace=True)
    X_input = df.to_numpy()
    Y_input = y_df.to_numpy(dtype='float64')
    Y_input = Y_input.reshape(Y_input.shape[0],1)
    X_input = np.insert(X_input,0,np.ones(X_input.shape[0]),axis=1)
    # t = np.dot(Y_input.T,Y_input)
    # print(X_input.shape)
    # print(Y_input)

    '''
    splitting the input for cross-validation to select best lambda
    '''

    k = 10
    xFolds = np.array_split(X_input,k)
    yFolds = np.array_split(Y_input,k)
    rparams = np.loadtxt(sys.argv[4])
    optl = 0
    min_err = np.inf

    for lmda in rparams:
        l2err = 0.0
        for i in range(k):
            #train and validation sets
            train = xFolds.copy()
            y_train = yFolds.copy()
            valid = xFolds[i]
            y_valid = yFolds[i]
            del train[i]
            del y_train[i]

            train = np.concatenate(train,axis=0)
            y_train = np.concatenate(y_train,axis=0)

            inv = np.dot(train.T,train)
            inv = inv + lmda*(np.identity(inv.shape[0]))
            h = np.dot(np.linalg.inv(inv),train.T)
            w = np.dot(h,y_train)

            #calculate L2 error
            err = np.dot(valid,w)-y_valid
            l2err += np.linalg.norm(err)/np.linalg.norm(y_valid)
        
        #print(l2err)
        if l2err<min_err:
            min_err=l2err
            optl = lmda
    
    #print(optl,min_err)
    '''
    Writing the best parameter value of lambda
    '''
    tf = open(sys.argv[7], "w")
    tf.write(str(optl))
    tf.close()
    #optl=10.0
    '''
    Final weights
    '''
    inv = np.dot(X_input.T,X_input)
    inv = inv + optl*(np.identity(inv.shape[0]))
    h = np.dot(np.linalg.inv(inv),X_input.T)
    w = np.dot(h,Y_input)
    np.savetxt(sys.argv[6],w)

    '''
    Processing test data
    '''

    df = pd.read_csv(sys.argv[3])
    df.drop(df.columns[0],axis=1,inplace=True)
    #print(df.head())
    X_test = df.to_numpy()
    X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)

    '''
    Final predictions on test data
    '''
    Y_test = np.dot(X_test,w)
    np.savetxt(sys.argv[5],Y_test)

elif sys.argv[1]=='c':

    df = pd.read_csv(sys.argv[2])
    df1 = pd.read_csv(sys.argv[3])
    df.drop(df.columns[0],axis=1,inplace=True)
    df1.drop(df1.columns[0],axis=1,inplace=True)

    #---------------------------------------------------------------------------------------------
    #TRAIN SET
    df.drop('Gender',axis=1,inplace=True)
    df.drop('Race',axis=1,inplace=True)
    df.drop('Ethnicity',axis=1,inplace=True)
    df.drop('Operating Certificate Number',axis=1,inplace=True)
    df.drop('Facility Id',axis=1,inplace=True)
    df.drop('Health Service Area',axis=1,inplace=True)
    df.drop('Zip Code - 3 digits',axis=1,inplace=True)
    df.drop('Payment Typology 1',axis=1,inplace=True)
    df.drop('Payment Typology 2',axis=1,inplace=True)
    df.drop('Payment Typology 3',axis=1,inplace=True)
    df.drop('CCS Diagnosis Code',axis=1,inplace=True)
    df.drop('CCS Procedure Code',axis=1,inplace=True)
    df.drop('APR DRG Code',axis=1,inplace=True)
    df.drop('APR MDC Code',axis=1,inplace=True)
    df.drop('APR Severity of Illness Code',axis=1,inplace=True)
    df.drop('Emergency Department Indicator',axis=1,inplace=True)

    df['Facility Name + Length of Stay'] = df['Facility Name']+1000*df['Length of Stay']

    #TEST SET
    df1.drop('Gender',axis=1,inplace=True)
    df1.drop('Race',axis=1,inplace=True)
    df1.drop('Ethnicity',axis=1,inplace=True)
    df1.drop('Operating Certificate Number',axis=1,inplace=True)
    df1.drop('Facility Id',axis=1,inplace=True)
    df1.drop('Health Service Area',axis=1,inplace=True)
    df1.drop('Zip Code - 3 digits',axis=1,inplace=True)
    df1.drop('Payment Typology 1',axis=1,inplace=True)
    df1.drop('Payment Typology 2',axis=1,inplace=True)
    df1.drop('Payment Typology 3',axis=1,inplace=True)
    df1.drop('CCS Diagnosis Code',axis=1,inplace=True)
    df1.drop('CCS Procedure Code',axis=1,inplace=True)
    df1.drop('APR DRG Code',axis=1,inplace=True)
    df1.drop('APR MDC Code',axis=1,inplace=True)
    df1.drop('APR Severity of Illness Code',axis=1,inplace=True)
    df1.drop('Emergency Department Indicator',axis=1,inplace=True)

    df1['Facility Name + Length of Stay'] = df1['Facility Name']+1000*df1['Length of Stay']


    #--------------TARGET ENCODING-----------------------
    ss = ['Facility Name',
    'Facility Name + Length of Stay',
    'Length of Stay',
    'Hospital County',
    'Age Group',
    'Type of Admission',
    'CCS Diagnosis Description',
    'CCS Procedure Description',
    'APR DRG Description',
    'APR MDC Description',
    'APR Severity of Illness Description',
    'APR Risk of Mortality',
    'Patient Disposition',
    'APR Medical Surgical Description',
    'Birth Weight']

    for s in ss:
        for ll in pd.unique(df[s]):
            m = df.loc[df[s]==ll]['Total Costs'].mean()
            df[s] = df[s].mask(df[s]==ll,m)
            df1[s] = df1[s].mask(df1[s]==ll,m)


    #print(df.head())
    #print(df1.head())

    y_df = df['Total Costs']
    #y_test = df1['Total Costs']
    #df1.drop('Total Costs',axis=1,inplace=True)
    df.drop('Total Costs',axis=1,inplace=True)
    X_input = df.to_numpy()
    X_test = df1.to_numpy()
    Y_input = y_df.to_numpy(dtype='float64')
    Y_input = Y_input.reshape(Y_input.shape[0],1)

    #--------------for error calc--------------------
    # Y_test = y_test.to_numpy(dtype='float64')
    # Y_test = Y_test.reshape(Y_test.shape[0],1)
    
    
    #-------------Polynomial Feature Add.-------------
    poly = PolynomialFeatures(2,interaction_only=False)
    X_input = poly.fit_transform(X_input)
    X_test = poly.fit_transform(X_test)
    #print(X_input.shape)
    #print(X_test.shape)

    #------------active features---------------------

    x = [False, True, True, False, True, False, True, True, True, True, True, True, True, True, True, True, False, 
    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, 
    True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, 
    True, True, True, False, True, True, True, True, True, True, True, True, True, False, True, True, True, True, 
    True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, 
    True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, 
    True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, 
    True, True, True, True, True, True, True, True, False, True, True]

    X_input = np.compress(x,X_input,axis=1)
    X_test = np.compress(x,X_test,axis=1)
    #print(X_input.shape)
    #print(X_test.shape)

    #------------Linear Regression--------------------

    inv = np.dot(X_input.T,X_input)
    h = np.dot(np.linalg.pinv(inv),X_input.T)
    w = np.dot(h,Y_input)

    Y_pred = np.dot(X_test,w)
    np.savetxt(sys.argv[4],Y_pred)
    
    #-----------err pred-----------------------------
    # err = (np.sum(np.square(Y_pred-Y_test)))
    # err = err/(np.sum(np.square(Y_test)))
    # print(err)
    
else:
    print('Invalid Mode Selected\nSelect a/b/c mode')