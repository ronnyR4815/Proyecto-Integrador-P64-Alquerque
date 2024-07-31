import pygame
import sys
import numpy as np
import joblib

# Cargar el modelo de ML desde el archivo .joblib
# Para probar la interfaz puede probarla online en el siguiente link:
# https://replit.com/join/egxyryoubp-miembro1819d
modelo = joblib.load('./4_modelos_movimientos/[modelo]perceptron/[modelo]MLPerceptron.joblib')

# Inicialización de Pygame
pygame.init()

# Configuración de la ventana
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 5, 5  # Número de filas y columnas del tablero
SQUARE_SIZE = WIDTH // COLS  # Tamaño de cada casilla del tablero

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Creación de la ventana
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Alquerque Game')

# Función para dibujar el tablero
def draw_board(board):
    window.fill(WHITE)  # Limpia la ventana con color blanco

    # Dibuja las líneas del tablero
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(window, BLACK, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)

    # Dibuja las fichas en el tablero según el estado actual
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] == 1:
                # Dibuja una "X" más gruesa para la ficha 1
                pygame.draw.line(window, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE // 4, row * SQUARE_SIZE + SQUARE_SIZE // 4), 
                                 (col * SQUARE_SIZE + 3 * SQUARE_SIZE // 4, row * SQUARE_SIZE + 3 * SQUARE_SIZE // 4), 7)
                pygame.draw.line(window, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE // 4, row * SQUARE_SIZE + 3 * SQUARE_SIZE // 4), 
                                 (col * SQUARE_SIZE + 3 * SQUARE_SIZE // 4, row * SQUARE_SIZE + SQUARE_SIZE // 4), 7)

            elif board[row][col] == 2:
                # Dibuja un círculo para la ficha 2
                pygame.draw.circle(window, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 3)

    pygame.display.update()  # Actualiza la ventana con los cambios

# Función para obtener la posición del clic
def get_click_pos(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

# Función para preparar los datos para el modelo de ML
def prepare_data(board, start_row, start_col, end_row, end_col, player):
    # Convertir el tablero en una lista de características
    board_features = []
    for row in board:
        board_features.extend(row)

    # Agregar las características del movimiento y el jugador
    data = board_features + [start_row, start_col, end_row, end_col, player]
    return np.array(data).reshape(1, -1)

# Función principal del juego
def main():
    board = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2]
    ]  # Ejemplo de estado inicial del tablero

    draw_board(board)  # Dibuja el tablero inicial

    selected_piece = None  # Variable para almacenar la pieza seleccionada
    current_player = 1  # Suponiendo que el jugador 1 comienza

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = get_click_pos(pos)
                print(f"Clicked on row {row}, col {col}")

                # Si no hay pieza seleccionada, intenta seleccionar una pieza
                if selected_piece is None:
                    if board[row][col] != 0:  # Si hay una ficha en la casilla clicada
                        selected_piece = (row, col)
                        print(f"Selected piece at row {row}, col {col}")
                else:
                    # Si ya hay una pieza seleccionada, intenta moverla a la casilla clicada
                    start_row, start_col = selected_piece
                    end_row, end_col = row, col
                    data = prepare_data(board, start_row, start_col, end_row, end_col, current_player)

                    # Realizar predicción
                    prediction = modelo.predict(data)
                    print(f"Model prediction: {prediction[0]}")

                    if prediction[0] == 1:
                        # Mover la ficha en el tablero
                        board[end_row][end_col] = board[start_row][start_col]
                        board[start_row][start_col] = 0
                        selected_piece = None  # Reinicia la selección de pieza
                    else:
                        # Anular el movimiento y reiniciar la selección de pieza
                        selected_piece = None

                    draw_board(board)  # Actualiza el tablero después de cada movimiento

                    # Alternar el turno del jugador
                    current_player = 1 if current_player == 2 else 2

if __name__ == "__main__":
    main()
