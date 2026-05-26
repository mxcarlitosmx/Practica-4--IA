import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset desde la carpeta actual
print("Cargando el archivo local iris.data...")
columnas = ['Largo_Sepalo', 'Ancho_Sepalo', 'Largo_Petalo', 'Ancho_Petalo', 'Clase']
df = pd.read_csv('iris.data', names=columnas)

print("\n--- REPORTE EDA (PUNTO 1) ---")

# 2. Número de filas y columnas
filas, columnas_totales = df.shape
print(f"Número de filas: {filas}")
print(f"Número de columnas: {columnas_totales}\n")

# 3. Distribución de clases
print("Distribución de clases:")
print(df['Clase'].value_counts().to_string(), "\n")

# 4. Estadística fundamental (Media y Desviación Estándar)
print("Estadística fundamental por atributo:")
estadisticas = df.describe().loc[['mean', 'std']]
print(estadisticas.to_string(), "\n")

# 5. Gráficas en 2D agrupadas por clases
print("Generando gráficas 2D... Cierra la ventana de las gráficas para terminar.")
sns.pairplot(df, hue='Clase', markers=["o", "s", "D"], palette="Set1")
plt.suptitle("Pares de atributos agrupados por clases (Dataset Iris)", y=1.02)
plt.show()
