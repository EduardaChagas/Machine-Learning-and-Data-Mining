"""
Created on Sat Aug 18 16:59:00 2018

@author: Eduarda Chagas
"""
import pandas as pd
import requests
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def readFile():
    data = pd.read_csv('abalone_dataset.csv')
    data_app = pd.read_csv('abalone_app.csv')
    return data, data_app

def clearData(data, data_app):
    data['sex'].replace(['M', 'F', 'I'],[0,1,2], inplace=True)
    data_app['sex'].replace(['M', 'F', 'I'],[0,1,2], inplace=True)
    data = data[data['height'] > 0]
    return data, data_app
    
def select(X, X_app, y, nk):
    selector = SelectKBest(chi2, k=nk).fit(X, y)
    X_new = selector.transform(X) 
    X_new_app = selector.transform(X_app) 
    scores = selector.scores_
    return X_new, X_new_app, scores

def defineScore(X_new, X_new_app, scores, myScore):
    j = 0
    for i in myScore:
        X_new[:,j] = X_new[:,j] * scores[i]
        X_new_app[:,j] = X_new_app[:,j] * scores[i]
        j = j + 1
    return X_new, X_new_app    

def normalization(X,strategy):
    if strategy == 1:
        X = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(X)
    else:
        X = MaxAbsScaler(copy=True).fit_transform(X) 
    return X

def SVMClassification(X_train,y_train,X_test):
    neigh = svm.SVC(kernel='rbf', gamma=10, C=5)
    neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    return y_pred

def SVMCrossValidation(X,y):    
    X = np.array(X)    
    y = np.array(y)
    result = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        neigh = svm.SVC(kernel='rbf', gamma=10, C=5)
        neigh.fit(X_train,y_train)
        y_pred = neigh.predict(X_test)
        result.append(neigh)
        print("Accuracy: ",accuracy_score(y_test, y_pred))
    return result

def KNNClassification(X_train,y_train,X_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    return y_pred

def knnCrossValidation(X,y):    
    X = np.array(X)  
    y = np.array(y)
    result = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train,y_train)    
        y_pred = neigh.predict(X_test)
        result.append(neigh)
        print("Accuracy: ",accuracy_score(y_test, y_pred))
    return result

def RFClassification(X_train,y_train,X_test):
    neigh = RandomForestClassifier(n_jobs=2, random_state=0)
    neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    return y_pred

def RFCrossValidation(X,y):    
    X = np.array(X)  
    y = np.array(y)
    result = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        neigh = RandomForestClassifier(n_jobs=2, random_state=0)
        neigh.fit(X_train,y_train)    
        y_pred = neigh.predict(X_test)
        result.append(neigh)
        print("Accuracy: ",accuracy_score(y_test, y_pred))
    return result

def NNClassification(X_train,y_train,X_test):
    neigh = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    return y_pred

def NNCrossValidation(X,y):    
    X = np.array(X)  
    y = np.array(y)
    result = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        neigh = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
        neigh.fit(X_train,y_train)    
        y_pred = neigh.predict(X_test)
        result.append(neigh)
        print("Accuracy: ",accuracy_score(y_test, y_pred))
    return result

def sendSolution(y_pred):
    URL = "https://aydanomachado.com/mlclass/03_Validation.php"    
    DEV_KEY = "Duma"    
    data = {'dev_key':DEV_KEY,'predictions':pd.Series(y_pred).to_json(orient='values')}
    r = requests.post(url = URL, data = data)
    print(" - Resposta do servidor:\n", r.text, "\n")

data, data_app = readFile()
data, data_app = clearData(data, data_app)

X = data.drop(['type'], axis=1)
y = data.type
X_app = data_app


X, X_app, scores = select(X, X_app, y, 8)
#array([409.80676458,  33.62167306,  30.59698414,  13.63884693,326.78697532, 118.84515474,  71.84747546, 105.06888866])
X, X_app = defineScore(X, X_app, scores, [1,2,3,4,5,6,7])

X = normalization(X, 1)
X_app = normalization(X_app, 1)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=39)

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
#result1 = knnCrossValidation(X,y)
#result2 = SVMCrossValidation(X,y)
#result3 = RFCrossValidation(X,y)
#result4 = NNCrossValidation(X,y)

y_pred = SVMClassification(X,y,X_app)
sendSolution(y_pred)