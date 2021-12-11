import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA



#-----------function for doing one hot encoding-----------------------------
def oneHotEnc(df,colname):
    enc = OneHotEncoder(handle_unknown='ignore')
    arr = enc.fit_transform(df[colname].to_numpy().reshape(-1,1)).toarray()
    print(enc.categories_)
    #print(arr)

    df.drop(colname,axis=1,inplace=True)
    df1 = pd.DataFrame(arr)

    for c in df1.columns:
        df[colname+str(c)] = df1[c]

    return df



#main function

df = pd.read_csv(sys.argv[1])
df.drop(df.columns[0],axis=1,inplace=True)


#--------------------------------------------------------------------------------------------
# One Hot Encoding on some features

# df = oneHotEnc(df,'Health Service Area')
# df = oneHotEnc(df,'Hospital County')
# df = oneHotEnc(df, 'Age Group')
# df = oneHotEnc(df, 'Gender')
# df = oneHotEnc(df,'Race')
# df = oneHotEnc(df,'Ethnicity')
# df = oneHotEnc(df, 'Emergency Department Indicator')

#---------------------------------------------------------------------------------------------
#Dropping some columns which have similar relation to total costs and some columns with very little relation to total costs

#--------------------------------------------#
#Gender Race and Ethnicity should have very little relation with total costs in ideal world
#Hence these features are tried with and without dropping them 

df.drop('Gender',axis=1,inplace=True)
df.drop('Race',axis=1,inplace=True)
df.drop('Ethnicity',axis=1,inplace=True)

#--------------------------------------------#
#Operating Certificate Number, Facility ID and Facility Name represents a single hospital 
#so any one of these three are kept for linear regression

df.drop('Operating Certificate Number',axis=1,inplace=True)
df.drop('Facility Id',axis=1,inplace=True)
#df.drop('Facility Name',axis=1,inplace=True)

#--------------------------------------------#
#Health Service Area, Hospital County and Zip Code represents a same locality of hospitals 
#so any one or two of these three depending on cross-validation scores are kept

df.drop('Health Service Area',axis=1,inplace=True)
#df.drop('Hospital County',axis=1,inplace=True)
df.drop('Zip Code - 3 digits',axis=1,inplace=True)

#--------------------------------------------#
#Ideally Payment typology should not effect the total cost for a patient so all of these three
#should be dropped. Cross-validation Score is also checked using any one of these three.

df.drop('Payment Typology 1',axis=1,inplace=True)
df.drop('Payment Typology 2',axis=1,inplace=True)
df.drop('Payment Typology 3',axis=1,inplace=True)

#--------------------------------------------#
#The following features are important in determining the total cost. Hence they are never dropped
#Age Group,
#Length of Stay,
#Type of Admission,
#Patient Disposition,
#APR Risk of Mortality

#--------------------------------------------#
#Birth Weight is tried with/without dropping
#df.drop('Birth Weight',axis=1,inplace=True)

#--------------------------------------------#
#There are features which have two cloumns- one for code and other for description.
#Hence only one of these two rows should be kept as the two columns are exactly the same thing
#Dropping all code columns for all description columns are tried.
#Dropping Code columns has less effect to total cost compared to Description columns

df.drop('CCS Diagnosis Code',axis=1,inplace=True)
df.drop('CCS Procedure Code',axis=1,inplace=True)
df.drop('APR DRG Code',axis=1,inplace=True)
df.drop('APR MDC Code',axis=1,inplace=True)
df.drop('APR Severity of Illness Code',axis=1,inplace=True)
df.drop('Emergency Department Indicator',axis=1,inplace=True)

# df.drop('CCS Diagnosis Description',axis=1,inplace=True)
# df.drop('CCS Procedure Description',axis=1,inplace=True)
# df.drop('APR DRG Description',axis=1,inplace=True)
# df.drop('APR MDC Description',axis=1,inplace=True)
# df.drop('APR Severity of Illness Description',axis=1,inplace=True)
# df.drop('APR Medical Surgical Description',axis=1,inplace=True)

#--------------------------------------------#
#Addition of combined columns:
#Due to high covariance between a column and total cost. 
#We try to encode the information of two cols into one col with higher covariance

df['Facility Name + Length of Stay'] = df['Facility Name']+1000*df['Length of Stay']
#df['CCS Description'] = df['CCS Procedure Description']+1000*df['CCS Diagnosis Description']

#--------------TARGET ENCODING--------------------
#The remaining features are target encoded as they represent qualitative classes
#Target encoding is done using pandas
#ss is the array of the remaining features

ss = ['Facility Name',
'Facility Name + Length of Stay',
'Hospital County',
'Age Group',
'Length of Stay',
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
        #print(ll)
        m = df.loc[df[s]==ll]['Total Costs'].mean()
        df[s] = df[s].mask(df[s]==ll,m)
        
#print(df.head())

#Input Y vector and Input X matrix are generted in np.array format
y_df = df['Total Costs']
df.drop('Total Costs',axis=1,inplace=True)
X_input = df.to_numpy()
Y_input = y_df.to_numpy(dtype='float64')
Y_input = Y_input.reshape(Y_input.shape[0],1)


#--------------PCA--------------------------------
#PCA is mainly used for reducing no of features but 
#PCA is tried for normalizing constants present in each column
#PCA has minimal effect so it is not used further

# pca = PCA()
# X_input = pca.fit_transform(X_input)
# print(X_input.shape)


#-------------Polynomial Feature Add.-------------
#The main gain in the Cross-validation score is due to addition of poly features
#Poly features upto degree 2 are added to the Input X matrix

poly = PolynomialFeatures(2,interaction_only=False)
X_input = poly.fit_transform(X_input)
#print(X_input.shape)


#cross validation with new features
k = 10
xFolds = np.array_split(X_input,k)
yFolds = np.array_split(Y_input,k)
rparams = np.array([0.000001,0.00001,0.0001,0.001,0.01,0.1])
optl = 0
maxss = 0.0
l = []
par = []

for lmda in rparams:
    sc = 0.0
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

        reg = linear_model.LassoLars(alpha=lmda,max_iter=500)
        reg.fit(train,y_train)

        par.append((lmda,i,len(reg.active_)))        
        sc += reg.score(valid,y_valid)

    sc = sc*10
    print(lmda,sc)
    l.append((lmda,sc))
    if sc>=maxss:
        maxss = sc
        optl = lmda

print('-----------------')
print(optl,maxss)


reg = linear_model.LassoLars(alpha=optl,max_iter=500)
reg.fit(X_input,Y_input)

np.savetxt('coeff.txt',reg.coef_)

#selecting active features
x = reg.coef_
l = []
for i in range(0,x.shape[0]):
    if(x[i]!=0.0):
        l.append(True)
    else:
        l.append(False)
print(l)

print(len(l))
c=0
for x in l:
    if x == True:
        c+=1
print(c)