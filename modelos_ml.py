# -*- coding: utf-8 -*-
"""Modelos_ML.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PlAi9F55CBYA8pQTHvppWpBioj6x5LQA

**Predicting Severity:** Predicting the severity of accidents is particularly important because it allows for timely and appropriate responses. By assessing accident severity, responders can allocate resources, prioritize medical treatment, and dispatch appropriate personnel. Predictive models can take into account various factors such as road conditions, weather, vehicle type, and collision type to estimate the likelihood of severe outcomes, helping improve emergency response and medical care.

#### librerias
"""

import os
import re
import string
import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import h2o
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from math import radians, sin, cos, sqrt, atan2
from shapely.geometry import Polygon
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from statsmodels.tools.tools import add_constant

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier

# Definir la ruta a la carpeta Data
#Data_path = '../Data'

# Cargar los datos
#file_path = os.path.join(Data_path, 'US_Accidents_March23_v2.csv')
df = pd.read_csv('US_Accidents_March23_v2.csv')
df.dropna(inplace=True)

# Tomar una muestra aleatoria del 10% de los datos
sample_df = df.sample(frac=0.1, random_state=42)  # random_state asegura reproducibilidad

# Mostrar las primeras filas de la muestra
print("Primeras filas de la muestra (10% del conjunto de datos):")
print('filas / columnas', sample_df.shape)
print(sample_df.head())

def classification_task(model, X_train_scaled, y_train, X_test_scaled, y_test, predic, model_name):
    """
    Evalúa el rendimiento de un modelo de clasificación y devuelve un DataFrame con varias métricas.

    Parámetros:
    - model: objeto del modelo de clasificación entrenado.
    - X_train_scaled: conjunto de datos de entrenamiento escalado.
    - y_train: etiquetas verdaderas del conjunto de entrenamiento.
    - X_test_scaled: conjunto de datos de prueba escalado.
    - y_test: etiquetas verdaderas del conjunto de prueba.
    - predic: predicciones del modelo en el conjunto de prueba.
    - model_name: nombre del modelo, utilizado como índice en el DataFrame de resultados.

    Retorno:
    - perf_df: DataFrame con las métricas de rendimiento del modelo.
    """

    # Crear el DataFrame con las métricas de evaluación
    perf_df = pd.DataFrame({
        'Train_Score': model.score(X_train_scaled, y_train),         # Puntaje en el conjunto de entrenamiento
        'Test_Score': model.score(X_test_scaled, y_test),            # Puntaje en el conjunto de prueba
        'Precision_Score': precision_score(y_test, predic, average='weighted'),  # Precisión ponderada
        'Recall_Score': recall_score(y_test, predic, average='weighted'),        # Recall ponderado
        'F1_Score': f1_score(y_test, predic, average='weighted'),               # F1 Score ponderado
        'accuracy': accuracy_score(y_test, predic)                              # Exactitud global
    }, index=[model_name])

    return perf_df

# seleccion variable respuesta
X = sample_df.drop('Severity', axis=1)
y= sample_df['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

"""#### RandomForestClassifier"""

# definicion del modelo
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
Eval_Rf= classification_task(rf_classifier,X_train, y_train ,X_test ,y_test, y_pred ,'Random Forest')
Eval_Rf

"""#### XGBoost"""

(y_train-1).unique()

X_train = X_train.astype({"Visibility_Category_Clear": "category",
                          "is_diciembre_Yes": "category",
                          "Is_Rush_Hour_Yes": "category"})
X_test = X_test.astype({"Visibility_Category_Clear": "category",
                        "is_diciembre_Yes": "category",
                        "Is_Rush_Hour_Yes": "category"})

xgb_classifier = XGBClassifier(objective='multi:softmax', random_state=42, enable_categorical=True)
xgb_classifier.fit(X_train, y_train-1 )

y_pred = xgb_classifier.predict(X_test)
Eval_xgb=classification_task(xgb_classifier,X_train, y_train-1, X_test, y_test-1, y_pred,'XGBoost')
Eval_xgb

#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# defina el servidor para llevar el registro de modelos y artefactos
mlflow.set_tracking_uri('http://localhost:5000')
# registre el experimento
experiment = mlflow.set_experiment("Modelos_ML")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas.
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo.
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    n_estimators = 200
    max_depth = 6
    max_features = 4
    # Cree el modelo con los parámetros definidos y entrénelo
    rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rf.fit(X_train, y_train)
    # Realice predicciones de prueba
    predictions = rf.predict(X_test)

    # Registre los parámetros
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("max_feat", max_features)

    # Registre el modelo
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Cree y registre la métrica de interés
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    print(mse)
