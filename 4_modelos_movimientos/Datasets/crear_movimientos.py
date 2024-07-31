import pandas as pd

data = pd.read_csv('4_modelos_movimientos/Datasets/tableros.csv')
data = data.values.tolist()

columns = [(i, j) for i in range(5) for j in range(5)]
df = pd.DataFrame(data, columns=columns)

# Función para generar los movimientos
def generar_movimientos(df):
    movimientos = []
    for index, row in df.iterrows():
        for i in range(5):
            for j in range(5):
                if row[(i, j)] in [1, 2]:  # Solo considerar fichas 1 o 2
                    jugador = row[(i, j)]
                    # Generar todas las posiciones vecinas
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue  # Ignorar la posición actual
                            ni, nj = i + di, j + dj
                            # Verificar que la nueva posición esté dentro del tablero
                            if 0 <= ni < 5 and 0 <= nj < 5:
                                movimiento_valido = 1 if row[(ni, nj)] == 0 else 0
                                movimiento = row.values.tolist()
                                movimiento.extend([i, j, ni, nj, jugador, movimiento_valido])
                                movimientos.append(movimiento)
    return movimientos

# Generar movimientos
movimientos = generar_movimientos(df)

# Aumentar las nuevas columnas de movimientos
columnas_movimientos = [
    "pos_X_actual",
    "pos_Y_actual",
    "pos_X_nueva",
    "pos_Y_nueva",
    "jugador",
    "validez",
]
columns.extend(columnas_movimientos)

df_movimientos = pd.DataFrame(movimientos, columns=columns)
df_movimientos.to_csv('4_modelos_movimientos/Datasets/movimientos_vecinos.csv', index=False)
