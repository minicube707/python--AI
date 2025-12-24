import numpy as np
import pygame
import matplotlib.pyplot as plt
import os

from System.Deep_Neuron_Network import softmax, foward_propagation
from System.Manage_file import select_model, load_model

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

def research(grid, parametres):

    grid = grid.reshape((1, 64))
    activation = foward_propagation(grid.T, parametres)
    C = len(parametres) // 2

    # Prédiction des probabilités avec softmax
    probabilities = softmax(activation["A" + str(C)].T)[0]
    pred = np.argmax(probabilities)
    porcent = np.max(probabilities)

    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})

    # Affichage de l'image
    axs[0].imshow(grid.reshape((8, 8)), cmap="gray")
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


def lister_dossiers():
    # Récupère le chemin du répertoire courant
    repertoire_courant = os.getcwd()
    
    # Liste uniquement les dossiers
    dossiers = [d for d in os.listdir(repertoire_courant) if os.path.isdir(d)]
    
    if not dossiers:
        print("Aucun dossier trouvé dans le répertoire courant.")
        return None
    
    # Affiche les dossiers avec un numéro
    print("Dossiers disponibles :")
    for i, dossier in enumerate(dossiers, start=1):
        print(f"{i}. {dossier}")
    
    # Demande à l'utilisateur de choisir un dossier
    while True:
        try:
            choix = int(input("\nEntrez le numéro du dossier à choisir : "))
            if 1 <= choix <= len(dossiers):
                dossier_choisi = dossiers[choix - 1]
                print(f"\nVous avez choisi : {dossier_choisi}")
                return dossier_choisi
            else:
                print("Numéro invalide, réessayez.")
        except ValueError:
            print("Veuillez entrer un nombre valide.")


#Main algorithm
def main (win , width):

    rows = 8
    grid = np.zeros((rows, rows))

    dir_name = lister_dossiers() 
    model, model_info = select_model(dir_name, "LogBook/model_logbook.csv")
    parametres = load_model(dir_name, model)
    
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
                    research(grid, parametres)
                
                if event.key == pygame.K_c:
                    grid = np.zeros((rows, rows))

        grid = add_node (width, rows, grid)
        grid = delete_node (width, rows, grid)
        draw(win, rows, width, grid)


main(WIN, WIDTH)