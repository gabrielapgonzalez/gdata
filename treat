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
description = pd.read_csv("/home/shink/gdata/hcg_gabi_shin/HomeCredit_columns_description.csv", encoding = "ISO-8859-1")
train.columns = map(str.lower, train.columns)
description.columns = map(str.lower, description.columns)

#tratamento
#cat_without_nulls - label encoder
#cat_with_nulls - se tem mais de de 30% nulo, fillna com moda
#cont_without_nulls - standart scaler
#cont_with_nulls - se tem mais de 30% nulo dropa, se nao fillna mean

def cat_with_nulls (train):
    for coluna in train.columns:
        if train[coluna].isnull().sum()/train.sk_id_curr.count() > 0.3:
            train = train.drop(coluna, axis=1)
        if str(train[coluna].dtype) == 'object' or 'int64':
            train = train[coluna].fillna(train[coluna].mode())
            train[coluna] = LabelEncoder().fit_transform(train[coluna])
        if str(train[coluna].dtype) == 'float64':
            train = train[coluna].fillna(train[coluna].mean())
            train[coluna] = Normalizer().fit_transform([train[coluna]])
    return train

df_train = train
