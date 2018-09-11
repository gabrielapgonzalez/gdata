import numpy as np
import pandas as pd
import logging
import io
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

os.chdir('/dados/home-credit')

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s;%(levelname)s;%(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger('model_hcg')


train = np.load(ptrain.py)
teste = np.load(ptest.py)

def prep_data(train = train):

	#Preparando o train
	X = train.drop("target", axis = 1)
	Y = train["target"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
	
	return  X_train, X_test, y_train, y_test

def run_model(model = "forest", X_train = X_train, y_train = y_train):

	if model == "forest":
	    rf = RandomForestClassifier(n_estimators = 500, criterion = "entropy", random_state=101)
	    rf.fit(X_train, y_train)
	    
    	return rf
		
def predict_md(md = md, X = teste):
	y_pred = md.predict(X)
	return y_pred
	

def metricas(metrica, md, y_pred, y_test):
	
	"""
	IMPUT
	
	metrical: "acc", "confussion"
	md: modelo
	y_pred: target predict
	y_test: target real
	"""
	
	if metrica = "acc"
		score = accuracy_score(y_test, y_pred)
		return score	
	#if metrica = "confussion"



X_train, X_test, y_train, y_test = prep_data()
md = run_model()
y = predict_md()

#Prep para submissão - falta renomear a coluna de TARGET
y = y.reset_index()
dic = {"sub": y}

np.save("y_sub.npy", dic)