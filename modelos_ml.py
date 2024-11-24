import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
print("Starting...")

df = pd.read_csv("US_Accidents_March23_v2.csv")
df.dropna(inplace=True)
sample_df = df.sample(frac=0.1, random_state=42)
X = sample_df.drop("Severity", axis=1)
y = sample_df["Severity"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# defina el servidor para llevar el registro de modelos y artefactos
mlflow.set_tracking_uri("http://localhost:8050")
experiment = mlflow.set_experiment("Modelos_ML")


with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    n_estimators = 200
    max_depth = 6
    max_features = 4
    # Cree el modelo con los parámetros definidos y entrénelo
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features
    )
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
