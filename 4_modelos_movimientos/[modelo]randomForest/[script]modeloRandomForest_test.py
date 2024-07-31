import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from joblib import load

# Cargar los datos
data = pd.read_csv('4_modelos_movimientos/ParticionarDataset/mov_test.csv')

# Separar dataset y etiquetas
X_test = data.drop(columns=['validez'])
y_test = data['validez']

# Cargar el modelo entrenado
model = load('4_modelos_movimientos/[modelo]randomForest/[modelo]RandomForest.joblib')

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Resumen del clasificador:\n{report}')