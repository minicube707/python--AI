import os
import numpy as np
import matplotlib
import pygame

import matplotlib.pyplot as plt

from System.Manage_file import select_model, load_model
from System.User_Input import handle_key

from System.Constante import FOLDER_NAME_LOGBOOK

from utilsConv import lister_dossiers, picture_prediction

matplotlib.use("TkAgg")  # Issue on linux PC 42

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("DeepNum")

BLACK =         (0, 0, 0)
GREY =          (128, 128, 128)
WHITE =         (255, 255, 255)

def draw_grid (win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))

        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, rows, width, grid):
    win.fill(BLACK)
    gap = width // rows

    for row in range(rows):
        for col in range(rows):
            value = grid[row, col]
            if value > 0:
                # Convertit la valeur (entre 0 et 1) en intensité de gris (0 à 255)
                color = (value, value, value)
                pygame.draw.rect(win, color, (col * gap, row * gap, gap, gap))

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos (pos, rows, width):

    x, y = pos
    gap = width // rows
    col = x // gap
    row = y // gap

    return row, col


def add_node(width, rows, grid, brush_size=2):
    # Left click
    if pygame.mouse.get_pressed()[0]:
        pos = pygame.mouse.get_pos()
        if (0 <= pos[0] < width) and (0 <= pos[1] < width):
            center_row, center_col = get_clicked_pos(pos, rows, width)
            max_intensity = 255

            for dr in range(-brush_size, brush_size + 1):
                for dc in range(-brush_size, brush_size + 1):
                    r = center_row + dr
                    c = center_col + dc

                    if 0 <= r < rows and 0 <= c < rows:
                        distance = np.sqrt(dr**2 + dc**2)
                        if distance <= brush_size:
                            intensity = max(0, int(max_intensity * (1 - (distance / brush_size))))
                            grid[r, c] = max(grid[r, c], intensity)  # Pour éviter d’écraser un plus fort dégradé
    return grid


def delete_node (width, rows, grid):

    #Right click
    if pygame.mouse.get_pressed()[2]:
        pos = pygame.mouse.get_pos()
        if (0 <= pos[0] <= width)  and (0 <= pos[1] <= width):
            row, col = get_clicked_pos(pos, rows, width)
            if (grid[row, col] != 0):
                grid[row, col] = 0
            

    return grid


def pooling(grid, kernel_size):

    # Nombre de blocs dans chaque dimension
    out_shape = (grid.shape[0] // kernel_size, grid.shape[1] // kernel_size)

    # Initialisation de la matrice résultat
    new_grid = np.zeros(out_shape)

    # Max pooling manuel avec un pas de kernel_size
    for i in range(0, grid.shape[0], kernel_size):
        for j in range(0, grid.shape[1], kernel_size):
            new_grid[i // kernel_size, j // kernel_size] = np.mean(grid[i:i + kernel_size, j:j + kernel_size])

    return new_grid


def research(grid, model, rows, hyperparams, do_pool):

    if (do_pool):
        grid = pooling(grid, kernel_size=2)
        rows = int(rows / 2)
        
    # =========================
    # Preprocessing
    # =========================
    grid /= 255
    img = grid[None, None, :, :]
    
    # =========================
    # Prediction
    # =========================
    prediction_scores, predicted_class, confidence_score = picture_prediction(model, hyperparams, img)
    
    # =========================
    # Display
    # =========================
    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Connecte l'événement clavier

    # Affichage de l'image
    axs[0].imshow(grid, cmap="gray")
    axs[0].set_title(f"Predict:{predicted_class} ({np.round(confidence_score, 2)}%)")
    axs[0].axis("off")

    # Affichage de l'histogramme des probabilités
    axs[1].bar(range(len(prediction_scores)), prediction_scores, color="blue")
    axs[1].set_xticks(range(len(prediction_scores)))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


#Main algorithm
def main (win, width):

    module_dir = lister_dossiers() 
    model_name, _ = select_model(module_dir, FOLDER_NAME_LOGBOOK)
    model, hyperparams, _, _, _, _ = load_model(module_dir, model_name, None)

    rows = hyperparams.input_shape[1]
    brush_size = int(input("What is the brush size ?\n"))
    str_pooling = input("Do want to use pooling ?\n").strip()

    do_pool = False
    if (str_pooling in {"true", "y", "yes"}):
        rows*=2
        do_pool = True
        
    grid = np.zeros((rows, rows))

    run = True
    while run:
        #Pygame event
        for event in pygame.event.get():

            #Quit pygame
            if event.type == pygame.QUIT:
                run = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False

                if event.key == pygame.K_SPACE:
                    research(grid.copy(), model, rows, hyperparams, do_pool)
                
                if event.key == pygame.K_c:
                    grid = np.zeros_like(grid)

        grid = add_node (width, rows, grid, brush_size)
        grid = delete_node (width, rows, grid)
        draw(win, rows, width, grid)


main(WIN, WIDTH)