import pickle
import numpy as np
import pygame
import matplotlib.pyplot as plt
import os

from File_Management import select_model,load_model

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


def draw (win, rows, width, grid):
    win.fill(BLACK)
    gap = width // rows

    coordinates = np.where(grid == 1)
    coordinates = np.vstack((coordinates[1], coordinates[0])).T
    for coord in coordinates:
        pygame.draw.rect(win, WHITE, (coord[0]*gap, coord[1]*gap,  gap, gap))

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos (pos, rows, width):

    x, y = pos
    gap = width // rows
    col = x // gap
    row = y // gap

    return row, col


def add_node (width, rows, grid):

    #Left click
    if pygame.mouse.get_pressed()[0]:
        pos = pygame.mouse.get_pos()
        if (0 <= pos[0] < width)  and (0 <= pos[1] < width):
            row, col = get_clicked_pos(pos, rows, width)
            if (grid[row, col] == 0):
                grid[row, col] = 1

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

def view_data(grid):

    plt.figure()
    plt.imshow(grid, cmap="gray")
    plt.grid("off")
    plt.show()


def display_data(data, target, max_par_fig=12):
    
    total = len(target)
    print(f"Data set size = {total}")
    data = data.reshape(-1, 8, 8)

    for start in range(0, total, max_par_fig):
        end = min(start + max_par_fig, total)
        batch = data[start:end]

        n = batch.shape[0]
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(cols * 4, rows * 3))
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(batch[i], cmap='gray')
            plt.title(target[start + i])
            plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Laisser de l’espace pour le suptitle
        plt.show()


def supprimer_grilles_dupliquees(data, target):

    new_data = np.array([])
    new_target = np.array([])
    vues = set()
    nb_doublons = 0

    data = data.reshape(-1, 8, 8)
    for i in range(len(target)):

        # On transforme la grille en une vue immuable (tuple)
        cle = tuple(data[i].flatten())
        if cle not in vues:
            vues.add(cle)
            new_data = np.append(new_data, data[i])
            new_target = np.append(new_target, target[i])
        else:
            nb_doublons += 1

    print(f"{nb_doublons} trouves")
    return new_data, new_target


#Main algorithm
def main (win , width):

    rows = 8
    grid = np.zeros((rows, rows))
    
    run = True
    if os.path.exists("digit_data.npz"):
        donnees = np.load("digit_data.npz")
        data_feature = donnees['data']
        data_target = donnees['targuet']
        print("Données chargées.")
    else:
        # Initialiser les arrays si pas de fichier trouvé
        data_feature = np.array([])
        data_target = np.array([])
        print("Fichier non trouvé, arrays initialisés.")
    

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
                    view_data(grid)
                
                if event.key == pygame.K_p:
                    data_feature = np.append(data_feature, grid)

                    number = input("What is the number ?\n")
                    data_target = np.append(data_target, np.int8(number))
                    grid = np.zeros((rows, rows))

                if  event.key == pygame.K_s:
                    data_feature, data_target = supprimer_grilles_dupliquees(data_feature, data_target)
                    display_data(data_feature, data_target)

                if event.key == pygame.K_c:
                    grid = np.zeros((rows, rows))

        grid = add_node (width, rows, grid)
        grid = delete_node (width, rows, grid)
        draw(win, rows, width, grid)
    
    np.savez("digit_data.npz", data=data_feature, targuet=data_target)


main(WIN, WIDTH)