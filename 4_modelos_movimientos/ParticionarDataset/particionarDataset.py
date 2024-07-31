import pandas as pd
from sklearn.model_selection import train_test_split

seed = 77

# Cargar los datos
data = pd.read_csv('4_modelos_movimientos/Datasets/movimientos_totales.csv')

# Randomizacion
data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

# Separar dataset y etiquetas
X = data.drop(columns=['validez'])
y = data['validez']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Combinar dataset y las etiquetas para guardar las particiones
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('4_modelos_movimientos/ParticionarDataset/mov_train.csv', index=False)
test_data.to_csv('4_modelos_movimientos/ParticionarDataset/mov_test.csv', index=False)