import numpy as np
from collections import Counter


# 1. FUNCIONES DE DISTANCIA
def distancia_euclidiana(pto1, pto2):
    return np.sqrt(np.sum((pto1 - pto2)**2))

def distancia_manhattan(pto1, pto2):
    return np.sum(np.abs(pto1 - pto2))

def distancia_chebyshev(pto1, pto2):
    return np.max(np.abs(pto1 - pto2))

# Diccionario para facilitar la elección del usuario
DISTANCIAS = {
    'euclidiana': distancia_euclidiana,
    'manhattan': distancia_manhattan,
    'chebyshev': distancia_chebyshev
}



# 2. CLASIFICADORES

# A. Clasificador Euclidiano (Distancia Mínima al Centroide)
def entrenar_centroides(X_train, y_train):
    """Calcula el punto promedio (centroide) de cada clase."""
    centroides = {}
    clases = np.unique(y_train)
    for c in clases:
        puntos_clase = X_train[y_train == c]
        centroides[c] = np.mean(puntos_clase, axis=0)
    return centroides

def clasificador_centroide(X_train, y_train, X_test, tipo_distancia='euclidiana'):
    centroides = entrenar_centroides(X_train, y_train)
    funcion_distancia = DISTANCIAS[tipo_distancia]
    y_pred = []
    
    for punto in X_test:
        # Calcula la distancia del punto a cada centroide
        distancias_centro = {c: funcion_distancia(punto, centro) for c, centro in centroides.items()}
        # Se queda con la clase del centroide más cercano
        clase_ganadora = min(distancias_centro, key=distancias_centro.get)
        y_pred.append(clase_ganadora)
        
    return np.array(y_pred)

# B. Clasificador K-Nearest Neighbors (KNN) y 1NN
def clasificador_knn(X_train, y_train, X_test, k, tipo_distancia='euclidiana'):
    funcion_distancia = DISTANCIAS[tipo_distancia]
    y_pred = []
    
    for punto in X_test:
        # 1. Medir distancia contra TODOS los puntos de entrenamiento
        distancias = [funcion_distancia(punto, x_train) for x_train in X_train]
        
        # 2. Obtener los índices de los K puntos más cercanos
        indices_cercanos = np.argsort(distancias)[:k]
        
        # 3. Obtener las clases de esos puntos cercanos
        clases_cercanas = [y_train[i] for i in indices_cercanos]
        
        # 4. Votación por mayoría (el más común)
        clase_ganadora = Counter(clases_cercanas).most_common(1)[0][0]
        y_pred.append(clase_ganadora)
        
    return np.array(y_pred)

# ==========================================
# 3. PRUEBA DE FUNCIONAMIENTO LOCAL
# ==========================================
if __name__ == "__main__":
    import pandas as pd
    
    print("Realizando prueba rápida de los clasificadores desde cero...")
    columnas = ['Largo_Sepalo', 'Ancho_Sepalo', 'Largo_Petalo', 'Ancho_Petalo', 'Clase']
    df = pd.read_csv('iris.data', names=columnas)
    
    # Separar atributos (X) y etiquetas (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Tomamos una flor de prueba arbitraria (ej. la primera fila) 
    flor_prueba = np.array([X[0]]) #de 0 a 49 setosa, de 50 a 99 versicolor, de 100 a 149 virginicas
    
    # Probando Clasificador Centroide
    pred_cent = clasificador_centroide(X, y, flor_prueba, 'manhattan')
    print(f"Predicción Clasificador Euclidiano (Manhattan): {pred_cent[0]}")
    
    # Probando 1NN
    pred_1nn = clasificador_knn(X, y, flor_prueba, k=1, tipo_distancia='chebyshev')
    print(f"Predicción 1NN (Chebyshev): {pred_1nn[0]}")
    
    # Probando KNN con K=5
    pred_5nn = clasificador_knn(X, y, flor_prueba, k=5, tipo_distancia='euclidiana')
    print(f"Predicción 5NN (Euclidiana): {pred_5nn[0]}")
    print("\n¡Las funciones matemáticas están listas!")
