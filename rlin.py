import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read dataset from csv
dataset = pd.read_csv("datos_mezclados.csv")
print ("Total number of rows in dataset: {}\n".format(len(dataset)))
print(dataset.head())

# Features
features = ['Temperature','Humidity','Heat Index','CO','Smoke','etiqueta']
target = 'etiqueta'

x_train, x_test, y_train, y_test = train_test_split(dataset[features], dataset[target],
                                                    train_size=0.7, test_size=0.3, shuffle=True)

# Print samples after running train_test_split
print("X_train: {}, Y_train: {}".format(len(x_train), len(x_test)))

print("\n")

model = LinearRegression()
model.fit(x_train, y_train)

# ... (código previo)

print("Showing Performance Metrics for Linear Regression\n")

print("Training R^2 Score: {:.2f}".format(model.score(x_train, y_train)))
predicted = model.predict(x_test)
print("Testing Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, predicted)))
print("Testing R^2 Score: {:.2f}".format(r2_score(y_test, predicted)))

#######################

import matplotlib.pyplot as plt

# Realizar predicciones en el conjunto de pruebas
predicted = model.predict(x_test)

# Graficar la línea de predicción vs valores reales
plt.scatter(y_test, predicted)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Valores reales vs Predicciones')

# Agregar la línea de regresión ideal (y = x) para comparación
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

##############################

