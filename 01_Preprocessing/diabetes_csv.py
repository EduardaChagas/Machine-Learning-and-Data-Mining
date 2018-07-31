#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
import requests
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
#import numpy as np 
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler


print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')
data_app = pd.read_csv('diabetes_app.csv')

print(' - Criando X e y  as colunas consideradas basta alterar o array a seguir.')
#feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
feature_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']

X = data[feature_cols]
y = data.Outcome

X_app = data_app[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=39)

#PREPROCESSAMENTO
#Pode ser “mean”, “median” e “most_frequent”
imputer = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=0)
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)
#X = imputer.fit_transform(X)
X_app = imputer.fit_transform(X_app)


#NORMALIZAÇÃO DO DADO 
#X_test = preprocessing.normalize(X_test, norm='l2')
#X_train = preprocessing.normalize(X_train, norm='l2') 
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
#scaler = MaxAbsScaler(copy=True)
scaler.fit_transform(X_test)
scaler.fit_transform(X_train)

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)
print("Accuracy: ",accuracy_score(y_test, y_pred))
