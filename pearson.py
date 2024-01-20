import pandas as pd
import matplotlib.pyplot as plt

def calcular_correlaciones(file1, file2):
    # Cargar los archivos CSV en marcos de datos separados
    data_c = pd.read_csv(file1)
    data_r = pd.read_csv(file2)

    # Excluir las columnas CO y Smoke si existen en Data_c.csv
    if 'CO' in data_c.columns and 'Smoke' in data_c.columns:
        data_c = data_c.drop(['CO', 'Smoke'], axis=1)

    # Calcular las correlaciones de Pearson, Spearman y Kendall entre los conjuntos de datos
    pearson_corr = data_c.corrwith(data_r, method='pearson')
    spearman_corr = data_c.corrwith(data_r, method='spearman')
    kendall_corr = data_c.corrwith(data_r, method='kendall')

    # Graficar las correlaciones
    plt.figure(figsize=(10, 6))
    correlaciones = pd.concat([pearson_corr, spearman_corr, kendall_corr], axis=1)
    correlaciones.columns = ['Pearson', 'Spearman', 'Kendall']
    correlaciones.plot(kind='bar', alpha=0.75)
    plt.title('Correlaciones entre Data_c.csv y Data_r.csv')
    plt.xlabel('Variables')
    plt.ylabel('Coeficiente de correlación')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Llamar a la función y pasar los nombres de archivo como argumentos
calcular_correlaciones('Data_c.csv', 'Data_r.csv')
