import pandas as pd

df = pd.read_csv("4_modelos_movimientos/[modelo]randomForest/[prueba]seedAccuracy.csv")

accuracy_mediana = df['Accuracy'].median()

# Semilla promedio entre todas las evaluadas
row = df.iloc[(df['Accuracy'] - accuracy_mediana).abs().argsort()[:1]]

seed = row['Seed'].values[0]
accuracy = row['Accuracy'].values[0]

print(f'Mejor semilla: {seed} - Mejor Accuracy: {accuracy}')