import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Cargar dataset
datos = pd.read_csv("dataset_transmilenio.csv")

# Variables de entrada
X = datos[["Hora", "Pasajeros", "Distancia"]]

# Variable objetivo
y = datos["Tiempo"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo de aprendizaje supervisado
modelo = LinearRegression()

# Entrenamiento
modelo.fit(X_train, y_train)

# Evaluación
predicciones = modelo.predict(X_test)

error = mean_absolute_error(y_test, predicciones)

print("Error medio absoluto:", round(error, 2))

# Predicción manual
hora = float(input("Hora del viaje: "))
pasajeros = float(input("Cantidad de pasajeros: "))
distancia = float(input("Distancia en km: "))

nuevo_viaje = pd.DataFrame({
    "Hora": [hora],
    "Pasajeros": [pasajeros],
    "Distancia": [distancia]
})

resultado = modelo.predict(nuevo_viaje)

print("\nTiempo estimado del viaje:")
print(round(resultado[0], 2), "minutos")
