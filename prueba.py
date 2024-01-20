import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def predict_and_plot(indice_input):
    # Carga de datos
    data = pd.read_csv('datos_sin_ceros.csv')

    # Separación de características y etiquetas
    X = data[['indice']]
    y = data[['CO', 'Smoke', 'Temperature', 'Humidity', 'Heat Index']]

    # División de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)
    model.feature_names_out_ = X_train.columns

    # Predicción basada en el índice ingresado
    predicted_values = model.predict([[indice_input]])

    # Visualización de datos y predicciones
    plt.figure(figsize=(10, 6))
    features = ['CO', 'Smoke', 'Temperature', 'Humidity', 'Heat Index']
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.scatter(X, y[features[i]], color='blue', label='Datos reales')
        plt.plot(X, model.predict(X)[:, i], color='red', label='Regresión')
        plt.scatter(indice_input, predicted_values[0][i], color='green', label='Predicción')
        plt.annotate(f'{predicted_values[0][i]:.2f}', (indice_input, predicted_values[0][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.xlabel('Índice')
        plt.ylabel(features[i])
        plt.title(f'Predicción de {features[i]}')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Ingreso del índice desde la terminal
indice_input = float(input("Ingresa el índice para predecir: "))
predict_and_plot(indice_input)


#################333

'''
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def predecir_variables(valor_indice):
    # Lee el archivo CSV
    datos = pd.read_csv("datos_sin_ceros.csv")
    
    # Extrae 'indice' como variable independiente 'x'
    x = datos["indice"].values.reshape(-1, 1)
    
    # Nombres de las columnas a predecir
    columnas_a_predecir = ['Temperature', 'Humidity', 'Heat Index', 'CO', 'Smoke']
    
    for columna in columnas_a_predecir:
        # Extrae la columna específica como variable dependiente 'y'
        y = datos[columna].values
        
        # Escala las características de entrada
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x)
        
        while True:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)  # Modifica el tamaño de test si es necesario

            mlr = MLPRegressor(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(100, 100, 100, 100), shuffle=True, max_iter=3000)#shuffle=True #random_state=42
            mlr.fit(X_train, y_train)

            train_score = mlr.score(X_train, y_train)
            print(f'Score en el conjunto de entrenamiento para {columna}: {train_score}')

            if train_score > 0.95:
                break
        
        # Valor para predecir
        new_value = np.array([[valor_indice]])
        predicted_value = mlr.predict(scaler.transform(new_value))
        print(f"Predicción para {columna} en el índice de tiempo {valor_indice}: {predicted_value}")

# Obtén el valor del índice de tiempo desde la terminal
valor_de_indice = float(input("Ingresa el índice de tiempo para predecir (por ejemplo, 433): "))

# Llama a la función para predecir cada columna usando el valor ingresado
predecir_variables(valor_de_indice)
'''

##################### ESTANDARIZAR_TODO    #################
'''
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def predecir_variables(valor_indice):
    # Lee el archivo CSV
    datos = pd.read_csv("datos_sin_ceros.csv")
    
    # Extrae 'indice' como variable independiente 'x'
    x = datos["indice"].values.reshape(-1, 1)
    
    # Nombres de las columnas a predecir
    columnas_a_predecir = ['Temperature', 'Humidity', 'Heat Index', 'CO', 'Smoke']
    
    for columna in columnas_a_predecir:
        # Extrae la columna específica como variable dependiente 'y'
        y = datos[columna].values
        
        # Escala las características de entrada y las salidas
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_x.fit_transform(x)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Se escalan y se transforman en un arreglo 1D
        
        while True:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)  # Modifica el tamaño de test si es necesario

            mlr = MLPRegressor(solver='lbfgs', alpha=1.0, hidden_layer_sizes=(400, 400, 400, 400), shuffle=True, max_iter=3000)
            mlr.fit(X_train, y_train)

            train_score = mlr.score(X_train, y_train)
            print(f'Score en el conjunto de entrenamiento para {columna}: {train_score}')

            if train_score > 0.95:
                break
        
        # Valor para predecir
        new_value = np.array([[valor_indice]])
        predicted_value = mlr.predict(scaler_x.transform(new_value))
        predicted_value_rescaled = scaler_y.inverse_transform(predicted_value.reshape(-1, 1)).flatten()  # Inversión de la escala
        print(f"Predicción para {columna} en el índice de tiempo {valor_indice}: {predicted_value_rescaled}")

# Obtén el valor del índice de tiempo desde la terminal
valor_de_indice = float(input("Ingresa el índice de tiempo para predecir (por ejemplo, 433): "))

# Llama a la función para predecir cada columna usando el valor ingresado
predecir_variables(valor_de_indice)
'''
'''
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def predecir_variables(valor_indice):
    # Lee el archivo CSV
    datos = pd.read_csv("datos_sin_ceros.csv")
    
    # Extrae 'indice' como variable independiente 'x'
    x = datos["indice"].values.reshape(-1, 1)
    
    # Nombres de las columnas a predecir
    columnas_a_predecir = ['Temperature', 'Humidity', 'Heat Index', 'CO', 'Smoke']
    
    for columna in columnas_a_predecir:
        # Extrae la columna específica como variable dependiente 'y'
        y = datos[columna].values
        
        # Escala las características de entrada
        scaler_x = StandardScaler()
        X_scaled = scaler_x.fit_transform(x)
        
        # Escala las salidas para 'CO' y 'Smoke' usando MinMaxScaler
        if columna in ['CO', 'Smoke']:
            scaler_y = MinMaxScaler()
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        while True:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)  # Modifica el tamaño de test si es necesario

            mlr = MLPRegressor(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(400, 400, 400, 400), shuffle=True, max_iter=4000)
            mlr.fit(X_train, y_train)

            train_score = mlr.score(X_train, y_train)
            print(f'Score en el conjunto de entrenamiento para {columna}: {train_score}')

            if train_score > 0.95:
                break
        
        # Valor para predecir
        new_value = np.array([[valor_indice]])
        predicted_value = mlr.predict(scaler_x.transform(new_value))
        predicted_value_rescaled = scaler_y.inverse_transform(predicted_value.reshape(-1, 1)).flatten()  # Inversión de la escala
        print(f"Predicción para {columna} en el índice de tiempo {valor_indice}: {predicted_value_rescaled}")

# Obtén el valor del índice de tiempo desde la terminal
valor_de_indice = float(input("Ingresa el índice de tiempo para predecir (por ejemplo, 433): "))

# Llama a la función para predecir cada columna usando el valor ingresado
predecir_variables(valor_de_indice)
'''