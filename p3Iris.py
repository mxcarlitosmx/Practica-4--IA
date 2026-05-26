import numpy as np
import pandas as pd
# Importamos el archivo que creaste en el paso anterior
import punto2_clasificadores as clf

# ==========================================
# 1. MÉTODOS DE VALIDACIÓN DESDE CERO
# ==========================================
def calcular_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def hold_out_70_30(X, y, modelo_func, **kwargs):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    corte = int(0.7 * len(X))
    
    idx_train, idx_test = indices[:corte], indices[corte:]
    y_pred = modelo_func(X[idx_train], y[idx_train], X[idx_test], **kwargs)
    return calcular_accuracy(y[idx_test], y_pred)

def k_fold_cross_validation(X, y, k_folds, modelo_func, **kwargs):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    folds = np.array_split(indices, k_folds)
    accuracies = []
    
    for i in range(k_folds):
        idx_test = folds[i]
        idx_train = np.hstack(folds[:i] + folds[i+1:])
        y_pred = modelo_func(X[idx_train], y[idx_train], X[idx_test], **kwargs)
        accuracies.append(calcular_accuracy(y[idx_test], y_pred))
    return np.mean(accuracies)

def leave_one_out(X, y, modelo_func, **kwargs):
    accuracies = []
    for i in range(len(X)):
        idx_test = [i]
        idx_train = np.delete(np.arange(len(X)), i)
        y_pred = modelo_func(X[idx_train], y[idx_train], X[idx_test], **kwargs)
        accuracies.append(calcular_accuracy(y[idx_test], y_pred))
    return np.mean(accuracies)

# ==========================================
# 2. EJECUCIÓN Y GENERACIÓN DE TABLA
# ==========================================
if __name__ == "__main__":
    print("Cargando datos y ejecutando validaciones (esto puede tomar unos segundos)...\n")
    
    # Cargar datos
    columnas = ['Largo_Sepalo', 'Ancho_Sepalo', 'Largo_Petalo', 'Ancho_Petalo', 'Clase']
    df = pd.read_csv('iris.data', names=columnas)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Configuraciones a probar
    distancias = ['euclidiana', 'manhattan', 'chebyshev']
    clasificadores = [
        ('Euclidiano (Centroide)', clf.clasificador_centroide, {}),
        ('1NN', clf.clasificador_knn, {'k': 1}),
        ('KNN (K=3)', clf.clasificador_knn, {'k': 3}),
        ('KNN (K=5)', clf.clasificador_knn, {'k': 5}),
        ('KNN (K=7)', clf.clasificador_knn, {'k': 7}),
        ('KNN (K=9)', clf.clasificador_knn, {'k': 9}),
        ('KNN (K=11)', clf.clasificador_knn, {'k': 11})
    ]

    resultados = []

    for nombre_clf, func_clf, params in clasificadores:
        for dist in distancias:
            params['tipo_distancia'] = dist
            
            # Ejecutar validaciones
            acc_ho = hold_out_70_30(X, y, func_clf, **params)
            acc_kf = k_fold_cross_validation(X, y, 10, func_clf, **params)
            acc_loo = leave_one_out(X, y, func_clf, **params)
            
            resultados.append({
                'Clasificador': nombre_clf,
                'Distancia': dist.capitalize(),
                'Hold Out 70/30': f"{acc_ho:.4f}",
                '10-Fold CV': f"{acc_kf:.4f}",
                'Leave-One-Out': f"{acc_loo:.4f}"
            })

    # Crear y mostrar la tabla comparativa final con pandas
    df_resultados = pd.DataFrame(resultados)
    print("================ TABLA COMPARATIVA DE RESULTADOS (ACCURACY) ================")
    print(df_resultados.to_string(index=False))
    print("============================================================================")
