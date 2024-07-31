import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

seed = 77

# Cargar los datos
data = pd.read_csv('4_modelos_movimientos/ParticionarDataset/mov_train.csv')

# Separar dataset y etiquetas
X_train = data.drop(columns=['validez'])
y_train = data['validez']

# Crear el modelo Random Forest
model = RandomForestClassifier(n_estimators=250, random_state=seed)

# Entrenar el modelo
model.fit(X_train, y_train)

dump(model, '4_modelos_movimientos/[modelo]randomForest/[modelo]RandomForest.joblib')