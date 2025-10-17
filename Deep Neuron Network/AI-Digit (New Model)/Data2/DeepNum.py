import pickle
import numpy as np
import pygame
import matplotlib.pyplot as plt
import os
import sys

from pathlib import Path

# Ajouter le dossier parent de Data1/ (donc C:/) à sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from System.Mathematical_function import softmax
from System.Deep_Neuron_Network import foward_propagation_DNN
from System.File_Management import select_model, load_model
from System.Preprocessing import  handle_key

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
            if (grid[row, col] == 1):
                grid[row, col] = 0
            

    return grid

def research(grid, parametres, model_info):

    grid = grid.reshape((1, 784))
    grid /= 255
    
    parametres_DNN, dimensions_DNN = parametres
    alpha = model_info["alpha"]

    C_DNN = len(dimensions_DNN)

    activation_DNN = foward_propagation_DNN(grid.T,  parametres_DNN, dimensions_DNN, C_DNN, alpha)
    
    # Prédiction des probabilités avec softmax
    probabilities = softmax(activation_DNN["A" + str(C_DNN)]).flatten()
    pred = np.argmax(probabilities)
    porcent = np.max(probabilities)
    
    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Connecte l'événement clavier

    # Affichage de l'image
    axs[0].imshow(grid.reshape((28, 28)), cmap="gray")
    axs[0].set_title(f"Predict:{pred} ({np.round(porcent, 2)}%)")
    axs[0].axis("off")

    # Affichage de l'histogramme des probabilités
    axs[1].bar(range(len(probabilities)), probabilities, color="blue")
    axs[1].set_xticks(range(len(probabilities)))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


#Main algorithm
def main (win , width):

    rows = 28
    grid = np.zeros((rows, rows))

    model, model_info = select_model(module_dir, "model_logbook.csv")
    parametres = load_model(module_dir, model)

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
                    research(grid, parametres, model_info)
                
                if event.key == pygame.K_c:
                    grid = np.zeros((rows, rows))

        grid = add_node (width, rows, grid, 2)
        grid = delete_node (width, rows, grid)
        draw(win, rows, width, grid)


main(WIN, WIDTH)