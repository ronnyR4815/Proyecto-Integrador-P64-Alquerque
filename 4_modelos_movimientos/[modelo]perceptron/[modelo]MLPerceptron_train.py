import pandas as pd
from sklearn.neural_network import MLPClassifier
from joblib import dump

seed = 77

# Cargar los datos
data = pd.read_csv('4_modelos_movimientos/ParticionarDataset/mov_train.csv')

# Separar dataset y etiquetas
X_train = data.drop(columns=['validez'])
y_train = data['validez']

# Crear el modelo de Perceptrón Multicapa
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation="logistic", random_state=seed, verbose=True, solver="lbfgs", alpha=0.001)

# Entrenar el modelo
mlp.fit(X_train, y_train)

dump(mlp, '4_modelos_movimientos/[modelo]perceptron/[modelo]MLPerceptron.joblib')