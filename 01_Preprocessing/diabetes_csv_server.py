import pandas as pd
import requests
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')
data_app = pd.read_csv('diabetes_app.csv')

# Criando X and y par ao para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificaralgorítmo de aprendizagem de máquina.\
print(' - Criando X e y  as colunas consideradas basta alterar o array a seguir.')
#feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
feature_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']

X = data[feature_cols]
y = data.Outcome

X_app = data_app[feature_cols]

#Determinando o conjunto de dados teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=39)

#PREPROCESSAMENTO
imputer = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=0)
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)
#X = imputer.fit_transform(X)
X_app = imputer.fit_transform(X_app)

#Pode ser “mean”, “median” e “most_frequent”
#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
#X = imputer.fit_transform(X)

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
#neigh.fit(X, y)
neigh.fit(X_train,y_train)
#y_pred = neigh.predict(X_test)
#print("Accuracy: ",accuracy_score(y_test, y_pred))

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
#data_app = pd.read_csv('diabetes_app.csv')
y_pred = neigh.predict(X_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

DEV_KEY = "Duma"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")