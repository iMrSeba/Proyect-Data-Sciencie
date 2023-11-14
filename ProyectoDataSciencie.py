from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import resample
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score

def mostrarDatos(df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df.head())

def eliminar_columnas(df, columnas_a_eliminar):
    nuevo_df = df.copy()
    nuevo_df.drop(columnas_a_eliminar, axis=1, inplace=True)
    return nuevo_df

def eliminar_filas_Demencia_nulos(df):
    column = df['Demencia']
    df_1 = df.dropna(subset=['Demencia'])
    df_1 = df_1.drop('Demencia', axis=1)
    
    imputer = SimpleImputer(strategy='mean')
    df_imputado = pd.DataFrame(imputer.fit_transform(df_1), columns=df_1.columns)
    df_imputado['Demencia'] = column

    df_imputado = df_imputado.dropna(subset=['Demencia'])
    return df_imputado

def mostrar_nulos(df):
    nulos_por_columna = df.isna().sum()
    info_nulos = pd.DataFrame({'Columna': nulos_por_columna.index, 'Valores NaN': nulos_por_columna.values})
    print(info_nulos)

def seleccionar_caracteristicas(X, y, num_caracteristicas=35):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y_encoded)

    sfm = SelectFromModel(clf, prefit=True, max_features=num_caracteristicas)
    caracteristicas_seleccionadas = X.columns[sfm.get_support()]

    # Obtener la importancia de cada característica
    importancias = clf.feature_importances_

    # Visualizar la importancia de las características
    plt.figure(figsize=(10, 6))
    plt.barh(caracteristicas_seleccionadas, importancias[sfm.get_support()])
    plt.xlabel('Importancia')
    plt.title('Importancia de las Características Seleccionadas')
    plt.show()
    
    
    return caracteristicas_seleccionadas

def bootstrap_predicciones(nuevo_df, columna_objetivo, num_bootstrap_samples=1105):
    all_predictions = []

    # Seleccionar características fuera del bucle
    X_original = nuevo_df.drop(columna_objetivo, axis=1)
    y_original = nuevo_df[columna_objetivo].map({'No': 0, 'Probable': 1, 'Posible': 2})
    caracteristicas_seleccionadas = seleccionar_caracteristicas(X_original, y_original)

    for _ in range(num_bootstrap_samples):
        bootstrap_sample = resample(nuevo_df, replace=True, random_state=42)

        X = bootstrap_sample.drop(columna_objetivo, axis=1)[caracteristicas_seleccionadas]
        y = bootstrap_sample[columna_objetivo].map({'No': 0, 'Probable': 1, 'Posible': 2})

        clf = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=5)
        clf.fit(X, y)

        predictions = clf.predict(X_original[caracteristicas_seleccionadas])

        all_predictions.append(predictions)

    final_predictions = np.median(np.array(all_predictions), axis=0)

    return final_predictions

def leave_one_out_cross_validation(X, y, clf):
    loo = LeaveOneOut()
    scores = []
    caracteristicas_seleccionadas_loo = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Seleccionar características para este conjunto de datos
        caracteristicas_seleccionadas = seleccionar_caracteristicas(X_train, y_train)
        caracteristicas_seleccionadas_loo.append(caracteristicas_seleccionadas)

        # Ajustar el clasificador al conjunto de entrenamiento
        clf.fit(X_train[caracteristicas_seleccionadas], y_train)

        # Hacer predicciones en el conjunto de prueba
        predictions = clf.predict(X_test[caracteristicas_seleccionadas])

        # Calcular la precisión y almacenarla en la lista de puntuaciones
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)

    return scores, caracteristicas_seleccionadas_loo

filename = 'fonis-jbekios.sav'
df = pd.read_spss(filename)

# Columnas eliminando Demencia
columnas_eliminar1 = ['ID', 'Folio', 'Sexo', 'Edad', 'años_escolaridad', 'educacion', 'GDS_REC','GDS']
df_1 = eliminar_columnas(df, columnas_eliminar1)

df_1_imputado = eliminar_filas_Demencia_nulos(df_1)

columna_objetivo = 'Demencia'
# Utilizar la función bootstrap
predicciones_bootstrap = bootstrap_predicciones(df_1_imputado, columna_objetivo)

# Convertir las etiquetas reales a 0, 1, 2
etiquetas_reales = df_1_imputado[columna_objetivo].map({'No': 0, 'Probable': 1, 'Posible': 2})

# Calcular y mostrar la precisión del modelo bootstrap
precision_bootstrap = accuracy_score(etiquetas_reales, predicciones_bootstrap)
print(f'Precisión con bootstrap: {precision_bootstrap}')

# Calcular el recall
recall_bootstrap = recall_score(etiquetas_reales, predicciones_bootstrap, average='weighted')
print(f'Recall con bootstrap: {recall_bootstrap}')


# Utilizar la función leave_one_out_cross_validation
#scores_loo, caracteristicas_seleccionadas_loo = leave_one_out_cross_validation(df_1_imputado.drop(columna_objetivo, axis=1), etiquetas_reales, DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=5))

# Mostrar los resultados de la validación cruzada LOO
#print(f'Precisión media (LOO): {np.mean(scores_loo)}')
