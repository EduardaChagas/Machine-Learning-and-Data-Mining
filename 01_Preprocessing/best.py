import pandas as pd
import requests
import numpy as np 
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def readFile():
    data = pd.read_csv('diabetes_dataset.csv')
    data_app = pd.read_csv('diabetes_app.csv')
    return data, data_app

#“mean”, “median” e “most_frequent”
def preProcessing(X, myStrategy):
    imputer = Imputer(missing_values='NaN', strategy=myStrategy, axis=0, verbose=0)
    X = imputer.fit_transform(X)
    return X

def normalization(X,strategy):
    if strategy == 1:
        X = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(X)
    else:
        X = MaxAbsScaler(copy=True).fit_transform(X) 
    return X

def knnCrossValidation(X):    
    X = np.array(X)
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

def KNN(X_train,y_train,X_app,n_neighbors):
    neigh = KNeighborsClassifier(n_neighbors)
    neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_app)
    return y_pred

def sendSolution(y_pred):
    URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"    
    DEV_KEY = "Duma"    
    data = {'dev_key':DEV_KEY,'predictions':pd.Series(y_pred).to_json(orient='values')}
    r = requests.post(url = URL, data = data)
    print(" - Resposta do servidor:\n", r.text, "\n")
    
def select(X, X_app, y, nk):
    selector = SelectKBest(chi2, k=nk).fit(X, y)
    X_new = selector.transform(X) 
    X_new_app = selector.transform(X_app) 
    scores = selector.scores_
    print(scores)
    return X_new, X_new_app, scores

def defineScore(X_new, X_new_app, scores, myScore):
    j = 0
    for i in myScore:
        X_new[:,j] = X_new[:,j] * scores[i]
        X_new_app[:,j] = X_new_app[:,j] * scores[i]
        j = j + 1
    return X_new, X_new_app    

data, data_app = readFile()
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
X_app = data_app[feature_cols]
y = data.Outcome
X = preProcessing(X, 'most_frequent')
X_new, X_new_app, scores = select(X, X_app, y, 4)
#[ 56.78719827 966.08089449  10.27957012  47.94991797 451.37833585 73.48729994   4.15608261  72.56885168]
##########################################################3
myScore = [1,4,5,7]
##########################################################3
X_new, X_new_app = defineScore(X_new, X_new_app, scores, myScore)
X_new = normalization(X_new,1)
X_new_app = normalization(X_new_app,1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18)
result = knnCrossValidation(X_new)

#y_pred = KNN(X_train,y_train,X_app,3)
sendSolution(result[6].predict(X_new_app))
