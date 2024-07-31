import pygame
import sys

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
                pygame.draw.line(window, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE//4, row * SQUARE_SIZE + SQUARE_SIZE//4), 
                                 (col * SQUARE_SIZE + 3*SQUARE_SIZE//4, row * SQUARE_SIZE + 3*SQUARE_SIZE//4), 7)
                pygame.draw.line(window, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE//4, row * SQUARE_SIZE + 3*SQUARE_SIZE//4), 
                                 (col * SQUARE_SIZE + 3*SQUARE_SIZE//4, row * SQUARE_SIZE + SQUARE_SIZE//4), 7)

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
                    board[row][col] = board[start_row][start_col]
                    board[start_row][start_col] = 0
                    selected_piece = None  # Reinicia la selección de pieza

                draw_board(board)  # Actualiza el tablero después de cada movimiento

if __name__ == "__main__":
    main()