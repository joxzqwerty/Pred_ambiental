import pandas as pd
import matplotlib.pyplot as plt

def calcular_correlacion_pearson(file1, file2):
    # Cargar los archivos CSV en marcos de datos separados
    data_c = pd.read_csv(file1)
    data_r = pd.read_csv(file2)

    # Excluir las columnas CO y Smoke si existen en Data_c.csv
    if 'CO' in data_c.columns and 'Smoke' in data_c.columns:
        data_c = data_c.drop(['CO', 'Smoke'], axis=1)

    # Calcular la correlación de Pearson entre los conjuntos de datos
    pearson_corr = data_c.corrwith(data_r, method='pearson')

    # Imprimir la correlación de Pearson
    print("Correlación de Pearson entre Data_c.csv y Data_r.csv:")
    print(pearson_corr)

    # Graficar la correlación de Pearson
    plt.figure(figsize=(10, 6))
    pearson_corr.plot(kind='bar', alpha=0.75, color='b')
    plt.title('Correlación de Pearson entre Data_c.csv y Data_r.csv')
    plt.xlabel('Variables')
    plt.ylabel('Coeficiente de correlación de Pearson')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Llamar a la función y pasar los nombres de archivo como argumentos
calcular_correlacion_pearson('Data_c.csv', 'Data_r.csv')



##################COMPARACION DE ARCHIVOS##################################

# Cargar los archivos CSV en marcos de datos separados
data_c = pd.read_csv('Data_c.csv')
data_r = pd.read_csv('Data_r.csv')

# Comparar los DataFrames fila por fila
diferencias = []

# Obtener el mínimo número de filas entre ambos DataFrames
min_rows = min(len(data_c), len(data_r))

for i in range(min_rows):
    if not data_c.iloc[i].equals(data_r.iloc[i]):
        diferencias.append(f"Diferencia en la fila {i + 1}:")

        # Comprobar las diferencias por columna en la fila actual
        for col in data_c.columns:
            if data_c.iloc[i][col] != data_r.iloc[i][col]:
                diferencias.append(f"    {col}: {data_c.iloc[i][col]} != {data_r.iloc[i][col]}")

# Guardar las diferencias encontradas en un archivo CSV
output_file = 'diferencias.csv'
with open(output_file, 'w') as file:
    for diff in diferencias:
        file.write(f"{diff}\n")

print(f"Las diferencias se han guardado en el archivo '{output_file}'.")

######################MEZCLA DE ARCHIVOS##################################

# Cargar los archivos CSV en marcos de datos separados
data_c = pd.read_csv('Data_c.csv')
data_r = pd.read_csv('Data_r.csv')

# Concatenar ambos DataFrames
datos_mezclados = pd.concat([data_c, data_r], ignore_index=True)

# Agregar una nueva columna 'indice' con enumeración de datos
datos_mezclados.insert(0, 'indice', range(1, len(datos_mezclados) + 1))

# Mostrar la información del DataFrame combinado
print("Información del DataFrame combinado:")
print(datos_mezclados.info())

# Guardar el DataFrame combinado en un nuevo archivo CSV
datos_mezclados.to_csv('datos_mezclados.csv', index=False)
print("El archivo 'datos_mezclados.csv' ha sido creado con los datos combinados.")


################ETIQUETADO######################

# Cargar el DataFrame combinado
datos_mezclados = pd.read_csv('datos_mezclados.csv')

# Función para verificar altos niveles de CO o Smoke
def check_high_levels(row):
    if row['CO'] > 500 or row['Smoke'] > 150:
        return 1  # Altos niveles de CO o Smoke
    else:
        return 0  # Bajos niveles de CO o Smoke

# Agregar la nueva columna 'etiqueta' basada en los altos niveles de CO o Smoke
datos_mezclados['etiqueta'] = datos_mezclados.apply(check_high_levels, axis=1)

# Mostrar las primeras filas del DataFrame actualizado
print(datos_mezclados.head())

# Guardar el DataFrame actualizado en datos_mezclados.csv
datos_mezclados.to_csv('datos_mezclados.csv', index=False)
print("Se ha agregado la columna 'etiqueta' al archivo 'datos_mezclados.csv'.")


#################algoritmos de precision####################

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Read dataset from csv
dataset = pd.read_csv("datos_mezclados.csv")
print ("Total number of rows in dataset: {}\n".format(len(dataset)))
print(dataset.head())

# FeaturesHeat Index
# Features
features = ['Temperature','Humidity','Heat Index','CO','Smoke','etiqueta']
target = 'etiqueta'

X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset[target],train_size=0.7, test_size=0.3, shuffle=True)#shuffle=True #random_state=42


# Resto del código para entrenar y evaluar modelos, y calcular métricas...
# Inicialización de modelos
svm_model = SVC(kernel='rbf')
c45_model = DecisionTreeClassifier()
logistic_model = LogisticRegression(max_iter=1000)

# Entrenamiento de modelos
svm_model.fit(X_train, y_train)
c45_model.fit(X_train, y_train)
logistic_model.fit(X_train, y_train)

# Predicción en datos de prueba
y_pred_svm = svm_model.predict(X_test)
y_pred_c45 = c45_model.predict(X_test)
y_pred_logistic = logistic_model.predict(X_test)

# Calcular métricas para SVM
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Calcular métricas para C4.5
conf_matrix_c45 = confusion_matrix(y_test, y_pred_c45)
precision_c45 = precision_score(y_test, y_pred_c45)
recall_c45 = recall_score(y_test, y_pred_c45)
f1_c45 = f1_score(y_test, y_pred_c45)
accuracy_c45 = accuracy_score(y_test, y_pred_c45)

# Calcular métricas para Regresión Logística
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic)
recall_logistic = recall_score(y_test, y_pred_logistic)
f1_logistic = f1_score(y_test, y_pred_logistic)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

# Mostrar métricas de evaluación
print("Métricas para SVM:")
print("Matriz de Confusión:")
print(conf_matrix_svm)
print(f"Precisión: {precision_svm}")
print(f"Recall: {recall_svm}")
print(f"Medida F: {f1_svm}")
print(f"Exactitud: {accuracy_svm}")
# Repetir para C4.5 y Regresión Logística

# Mostrar métricas de evaluación para C4.5
print("Métricas para C4.5 (Árboles de Decisión):")
print("Matriz de Confusión:")
print(conf_matrix_c45)
print(f"Precisión: {precision_c45}")
print(f"Recall: {recall_c45}")
print(f"Medida F: {f1_c45}")
print(f"Exactitud: {accuracy_c45}")

# Mostrar métricas de evaluación para Regresión Logística
print("Métricas para Regresión Logística:")
print("Matriz de Confusión:")
print(conf_matrix_logistic)
print(f"Precisión: {precision_logistic}")
print(f"Recall: {recall_logistic}")
print(f"Medida F: {f1_logistic}")
print(f"Exactitud: {accuracy_logistic}")

######################matrices###################################
import seaborn as sns

# Función para graficar una matriz de confusión
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 5))
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicho')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# Calcular las matrices de confusión para SVM, C4.5 y Regresión Logística
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
conf_matrix_c45 = confusion_matrix(y_test, y_pred_c45)
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)

# Graficar las matrices de confusión para SVM, C4.5 y Regresión Logística
plot_confusion_matrix(conf_matrix_svm, 'Matriz de Confusión - SVM')
plot_confusion_matrix(conf_matrix_c45, 'Matriz de Confusión - Árboles de Decisión (C4.5)')
plot_confusion_matrix(conf_matrix_logistic, 'Matriz de Confusión - Regresión Logística')

##################resultados_predichos_sin ceros###########################

# Leer el archivo CSV
data = pd.read_csv("datos_mezclados.csv")

# Eliminar las filas que contienen ceros en alguna de las columnas
data = data[(data != 0).all(axis=1)]

# Guardar el nuevo conjunto de datos en un archivo CSV
data.to_csv('datos_sin_ceros.csv', index=False)

#######################ENTRENAMIENTO RED NEURONAL_NORMALIZAR CO Y SMOKE

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
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, shuffle=True)  # Modifica el tamaño de test si es necesario

            mlr = MLPRegressor(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(400, 400, 400, 400), random_state=42, max_iter=8000)
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

####################regresion linear
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Carga de datos
data = pd.read_csv('datos_sin_ceros.csv')

# Separación de características y etiquetas
X = data[['indice']]
y = data[['CO', 'Smoke', 'Temperature', 'Humidity', 'Heat Index']]

# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)
model.feature_names_out_ = X_train.columns


# Ingreso del índice desde la terminal
indice_input = float(input("Ingresa el índice para predecir: "))
indice_input = [[indice_input]]  # Convertir a formato adecuado para la predicción

# Predicción basada en el índice ingresado
predicted_values = model.predict(indice_input)

# Extracción de las predicciones para cada característica
predicted_CO = predicted_values[0][0]
predicted_Smoke = predicted_values[0][1]
predicted_Temperature = predicted_values[0][2]
predicted_Humidity = predicted_values[0][3]
predicted_Heat_Index = predicted_values[0][4]

# Visualización de datos y predicciones
plt.figure(figsize=(10, 6))

# Función para mostrar valores al apuntar el cursor
def annotate_plot(x, y, value):
    plt.annotate(f'{value:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Gráfico para CO
plt.subplot(2, 3, 1)
plt.scatter(X, y['CO'], color='blue', label='Datos reales')
plt.plot(X, model.predict(X)[:, 0], color='red', label='Regresión')
plt.scatter(indice_input, predicted_CO, color='green', label='Predicción')
annotate_plot(indice_input[0][0], predicted_CO, predicted_CO)
plt.xlabel('Índice')
plt.ylabel('CO')
plt.title('Predicción de CO')
plt.legend()

# Gráfico para Smoke
plt.subplot(2, 3, 2)
plt.scatter(X, y['Smoke'], color='blue', label='Datos reales')
plt.plot(X, model.predict(X)[:, 1], color='red', label='Regresión')
plt.scatter(indice_input, predicted_Smoke, color='green', label='Predicción')
annotate_plot(indice_input[0][0], predicted_Smoke, predicted_Smoke)
plt.xlabel('Índice')
plt.ylabel('Smoke')
plt.title('Predicción de Smoke')
plt.legend()

# Gráfico para Temperature
plt.subplot(2, 3, 3)
plt.scatter(X, y['Temperature'], color='blue', label='Datos reales')
plt.plot(X, model.predict(X)[:, 2], color='red', label='Regresión')
plt.scatter(indice_input, predicted_Temperature, color='green', label='Predicción')
annotate_plot(indice_input[0][0], predicted_Temperature, predicted_Temperature)
plt.xlabel('Índice')
plt.ylabel('Temperature')
plt.title('Predicción de Temperature')
plt.legend()

# Gráfico para Humidity
plt.subplot(2, 3, 4)
plt.scatter(X, y['Humidity'], color='blue', label='Datos reales')
plt.plot(X, model.predict(X)[:, 3], color='red', label='Regresión')
plt.scatter(indice_input, predicted_Humidity, color='green', label='Predicción')
annotate_plot(indice_input[0][0], predicted_Humidity, predicted_Humidity)
plt.xlabel('Índice')
plt.ylabel('Humidity')
plt.title('Predicción de Humidity')
plt.legend()

# Gráfico para Heat Index
plt.subplot(2, 3, 5)
plt.scatter(X, y['Heat Index'], color='blue', label='Datos reales')
plt.plot(X, model.predict(X)[:, 4], color='red', label='Regresión')
plt.scatter(indice_input, predicted_Heat_Index, color='green', label='Predicción')
annotate_plot(indice_input[0][0], predicted_Heat_Index, predicted_Heat_Index)
plt.xlabel('Índice')
plt.ylabel('Heat Index')
plt.title('Predicción de Heat Index')
plt.legend()

plt.tight_layout()
plt.show()
