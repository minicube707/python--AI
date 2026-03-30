
import numpy as np
import pygame
import matplotlib
matplotlib.use("TkAgg")  # Issue on linux PC 42
import matplotlib.pyplot as plt
import os

from System.Manage_file import select_model, load_model
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

def research(grid, model, rows, hyperparams, str_pooling):

    if (str_pooling == "true"):
        grid = pooling(grid, kernel_size=2)
        rows = int(rows / 2)

    grid /= 255
    
    # Prédiction des probabilités avec softmax
    y_pred = model.forward_propagation(grid[None, None, ...], False).flatten()

    pred = np.argmax(y_pred)
    porcent = np.max(y_pred)
    
    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Connecte l'événement clavier

    # Affichage de l'image
    axs[0].imshow(grid.reshape((rows, rows)), cmap="gray")
    axs[0].set_title(f"Predict:{pred} ({np.round(porcent, 2)}%)")
    axs[0].axis("off")

    # Affichage de l'histogramme des probabilités
    axs[1].bar(range(len(y_pred)), y_pred, color="blue")
    axs[1].set_xticks(range(len(y_pred)))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def lister_dossiers():
    # Récupère le chemin du répertoire courant
    repertoire_courant = os.getcwd()
    
    # Liste uniquement les dossiers qui contient des models
    dossiers = [
    d for d in os.listdir(repertoire_courant)
    if os.path.isdir(os.path.join(repertoire_courant, d)) and "Package" in d
    ]
            
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
            
            elif choix == 0:
                exit(1)

            else:
                print("Numéro invalide, réessayez.")

        except ValueError:
            print("Veuillez entrer un nombre valide.")


#Main algorithm
def main (win, width):

    module_dir = lister_dossiers() 
    model_name = select_model(module_dir, "LogBook")
    model, hyperparams, _, _, _, _ = load_model(module_dir, model_name)

    rows = hyperparams.input_shape[1]
    brush_size = int(input("What is the brush size ?\n"))
    str_pooling = input("Do want to use pooling ?\n")

    if (str_pooling == "true"):
        rows*=2
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
                    research(grid, model, rows, hyperparams, str_pooling)
                
                if event.key == pygame.K_c:
                    grid = np.zeros((rows, rows))

        grid = add_node (width, rows, grid, brush_size)
        grid = delete_node (width, rows, grid)
        draw(win, rows, width, grid)


main(WIN, WIDTH)