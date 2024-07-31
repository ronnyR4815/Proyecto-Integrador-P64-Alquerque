import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump, load

# Función para entrenar y evaluar el modelo con una semilla específica
def train_and_evaluate(seed, data):
    # Randomizacion
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    X = data.drop(columns=['validez'])
    y = data['validez']

    # Separar dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Crear el modelo Random Forest
    model = RandomForestClassifier(n_estimators=250, random_state=seed)
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Cargar los datos
data = pd.read_csv('4_modelos_movimientos/Datasets/movimientos_totales.csv')

results = []

for seed in range(1, 101):
    accuracy = train_and_evaluate(seed, data)
    results.append({'Seed': seed, 'Accuracy': accuracy})
    print(f'{seed} - {accuracy}')

# Convertir la lista de resultados a un DataFrame
results_df = pd.DataFrame(results)

# Guardar los resultados en un archivo CSV
results_df.to_csv('4_modelos_movimientos/[modelo]randomForest/[prueba]seedAccuracy.csv', index=False)

print(f'Resultados guardados en {results_df}')
