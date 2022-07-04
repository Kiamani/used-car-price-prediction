# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:21:06 2022

@author: asus
"""

import numpy as np
import pandas as pd
import scipy
from scipy.stats import zscore
import matplotlib.pyplot as plt
import sklearn 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import GridSearchCV
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('car.csv')
df.head()

df = pd.DataFrame(data=df)
df.tail()

df.shape

df.dtypes
df.info()
df.columns

#Checking for Null values
df.isnull().sum()

sn.heatmap(df.isnull())

from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df['mileage']=imp.fit_transform(df['mileage'].values.reshape(-1,1))
df['engine']=imp.fit_transform(df['engine'].values.reshape(-1,1))
df['torque']=imp.fit_transform(df['torque'].values.reshape(-1,1))
df['seats']=imp.fit_transform(df['seats'].values.reshape(-1,1))
df['max_power']=imp.fit_transform(df['max_power'].values.reshape(-1,1))

df.isnull().sum()

sn.boxplot(df['selling_price'])

sn.distplot(df['selling_price'], color="red")

collist=df.columns.values
plt.figure(figsize=(20,20))
for i in range(0,len(collist)):
    plt.subplot(3,5,i+1)
    sn.histplot(data=df[collist[i]],color='red').set_xticklabels(labels=df[collist[i]].unique(),rotation=90)
    plt.tight_layout()

df['owner'].info()
newcollist=['owner','transmission','fuel']
plt.figure(figsize=(20,20))
for i in enumerate(newcollist,1):
    plt.subplot(3,3,i[0])
    sn.barplot(data=df,x=i[1],y='selling_price',color='blue')
    plt.xticks(rotation=90)
    plt.tight_layout()



le = LabelEncoder()
for column in df.drop(['selling_price'],axis=1).columns:
    df[column]=le.fit_transform(df[column])
 

#heatmap 
plt.figure(figsize=(15,10))
sn.heatmap(round(df.describe()[1:].transpose(),2),lw=3,linecolor='red',annot=True,fmt='f',color='blue')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

#corr
plt.figure(figsize=(15,8))
sn.heatmap(df.corr(), linewidth=0.1, annot = False)

#removing price form columns
x= df.drop(['selling_price'],axis=1)
y=df['selling_price']

#standardizing
x=power_transform(x,method='yeo-johnson')
scale = StandardScaler()
x=scale.fit_transform(x)
x

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.4, random_state = 42)

#svr
svr = SVR()
svr.fit(xtrain,ytrain)
pred_train_svr=svr.predict(xtrain)
pred_test_svr=svr.predict(xtest)
print('SVR  Score:',svr.score(xtrain,ytrain))
print('SVR  r2_score:',r2_score(ytest,pred_test_svr))
print("Mean squared error of SVR :",mean_squared_error(ytest,pred_test_svr))


#reg
lr= LinearRegression()
lr.fit(xtrain,ytrain)
lr.coef_
pred_train=lr.predict(xtrain)
pred_test=lr.predict(xtest)
print('Linear Regression Score:',lr.score(xtrain,ytrain))
print('Linear Regression r2_score:',r2_score(ytest,pred_test))
print("Mean squared error of Linear Regression:",mean_squared_error(ytest,pred_test))



#SGD
sgd=SGDRegressor()
sgd.fit(xtrain,ytrain)
pred_train_sgd=sgd.predict(xtrain)
pred_test_sgd=sgd.predict(xtest)
print('SGD Score:',sgd.score(xtrain,ytrain))
print('SGD r2_score:',r2_score(ytest,pred_test_sgd))
print("Mean squared error SGD:",mean_squared_error(ytest,pred_test_sgd))


#KNeighbor
knr = KNeighborsRegressor()
knr.fit(xtrain,ytrain)
pred_train_knr=knr.predict(xtrain)
pred_test_knr=knr.predict(xtest)
print('K Neighbors  Score:',knr.score(xtrain,ytrain))
print('K Neighbors r2_score:',r2_score(ytest,pred_test_knr))
print("Mean squared error of K Neighbors :",mean_squared_error(ytest,pred_test_knr))


#Decision Tree
dtr=DecisionTreeRegressor(criterion='mse')
dtr.fit(xtrain,ytrain)
pred_train_dtr=dtr.predict(xtrain)
pred_test_dtr=dtr.predict(xtest)
print('Decision Tree  Score:',dtr.score(xtrain,ytrain))
print('Decision Tree  r2_score:',r2_score(ytest,pred_test_dtr))
print("Mean squared error of Decision Tree Regressor:",mean_squared_error(ytest,pred_test_dtr))


#Random Forest
rf=RandomForestRegressor()
rf.fit(xtrain,ytrain)
pred_train_rf=rf.predict(xtrain)
pred_test_rf=rf.predict(xtest)
print('Random Forest  Score:',rf.score(xtrain,ytrain))
print('Random Forest  r2_score:',r2_score(ytest,pred_test_rf))
print("Mean squared error of Random Forest Regressor:",mean_squared_error(ytest,pred_test_rf))


#Model
pricecar = RandomForestRegressor(bootstrap=False,min_samples_leaf=1,min_samples_split=2)
pricecar.fit(xtrain,ytrain)

a= np.array(ytest)
predicted = np.array(pricecar.predict(xtest))
Price=pd.DataFrame({"Original":a,"Predicted":predicted},index=range(len(a)))
Price
plt.plot(Price['Original'],Price['Predicted'])























