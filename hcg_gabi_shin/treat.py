import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

#Libraries setup
os.chdir('/dados/home-credit')

#Import and setup datasets
train = pd.read_csv('application_train.csv')
train.columns = map(str.lower, train.columns)

train_wk = train.drop(['target','sk_id_curr'], axis=1)
train_wk = train_wk.sample(n=1000)

#tratamento
#cat_without_nulls - label encoder
#cat_with_nulls - se tem mais de de 30% nulo, fillna com moda
#cont_without_nulls - standart scaler
#cont_with_nulls - se tem mais de 30% nulo dropa, se nao fillna mean

def tratamento (train):
    dp = []
    for ls in train.columns:
        if train[ls].isnull().sum()/train.count().max() > 0.3:
            dp.append(ls)
    train.drop(dp, axis=1, inplace=True)       
    for col in train.columns:
        if str(train[col].dtype) == 'float64':
            if train[col].value_counts().count() < 50:
                train[col] = train[col].fillna(train[col].mode().iloc[0]).astype('int64')
        if str(train[col].dtype) == 'int64':
            if train[col].value_counts().count() > 50:
                train[col] = train[col].fillna(train[col].mean()).astype('float64')            
        if str(train[col].dtype) == 'object':
            train[col] = train[col].fillna(train[col].mode().iloc[0])
            train[col] = LabelEncoder().fit_transform(train[col])
        if str(train[col].dtype) == 'int64':
            train[col] = train[col].fillna(train[col].mode().iloc[0])
            train[col] = LabelEncoder().fit_transform(train[col])            
    nc = []
    for ct in train.columns:
        if str(train[ct].dtype) == 'float64':
            nc.append(ct)
            train[ct] = train[ct].fillna(train[ct].mean())
#            train[ct] = Normalizer().fit_transform(train[nc])
    return train

df_treat = tratamento(train=train_wk)

#print(df_treat.columns)
print(df_treat.info())
print(df_treat.head())
print(df_treat.amt_credit.isnull().sum())
print(df_treat.isnull().any())