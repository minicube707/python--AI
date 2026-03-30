
import numpy as np
import matplotlib.pyplot as plt
from .Layer import Convolution
from .Preprocessing import handle_key

def display_comparaison_layer(A, Z=None, max_par_fig=12):
    """
    Affiche chaque couche du tableau 3D A, et optionnellement Z si fourni,
    côte à côte. S'adapte si Z est None.
    """
    if A.ndim == 4:
        A = A[0]

    elif A.ndim != 3:
        raise ValueError("A doit être un array 3D (D, H, W)")

    if Z is not None:
        
        if Z.ndim == 4:
            Z = Z[0]

        if Z.shape != A.shape:
            raise ValueError("A et Z doivent avoir la même forme si Z est fourni")
        mode_paire = True
    else:
        mode_paire = False

    total_couches = A.shape[0]

    for start in range(0, total_couches, max_par_fig):
        end = min(start + max_par_fig, total_couches)
        n = end - start

        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        total_subplots = cols * rows

        fig_cols = cols * 2 if mode_paire else cols
        fig, axes = plt.subplots(rows, fig_cols, figsize=(4 * cols, 3 * rows))
        fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche
        
        # Assurer que axes est toujours 2D
        if rows == 1:
            axes = np.expand_dims(axes, 0)
        if fig_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for i in range(n):
            layer_idx = start + i
            row = i // cols
            col = i % cols

            # Affichage de A
            ax_a = axes[row, col * 2] if mode_paire else axes[row, col]
            im_a = ax_a.imshow(A[layer_idx])
            ax_a.set_title(f"A - Couche {layer_idx}")
            ax_a.axis('off')
            fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)

            # Affichage de Z si présent
            if mode_paire:
                ax_z = axes[row, col * 2 + 1]
                im_z = ax_z.imshow(Z[layer_idx])
                ax_z.set_title(f"Z - Couche {layer_idx}")
                ax_z.axis('off')
                fig.colorbar(im_z, ax=ax_z, fraction=0.046, pad=0.04)

        # Masquer les axes inutilisés
        for j in range(n, total_subplots):
            row = j // cols
            col = j % cols
            if mode_paire:
                axes[row, col * 2].axis('off')
                axes[row, col * 2 + 1].axis('off')
            else:
                axes[row, col].axis('off')

        plt.suptitle(f'Couches {start} à {end - 1}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()



def display_activation(X, y, model):

    print("")
    number_wanted = int(input("Which number do want ?\n"))

    # Trouver tous les index correspondant au chiffre voulu
    indices = [i for i, label in enumerate(y) if label == number_wanted]

    # Choisir un index aléatoire parmi ceux-là
    index_choisi = np.random.choice(indices)

    X_chosen = X[index_choisi]

    # Afficher l'image X
    plt.imshow(X_chosen)
    plt.title(f"Chiffre: {y[index_choisi]}")
    plt.axis('off')
    plt.show()
    
    if X_chosen.ndim == 2:
        X_chosen = X_chosen[None, None, ...]

    if X_chosen.ndim == 3:
        X_chosen = X_chosen[None, ...].transpose(0, 3, 1, 2)

    C = model.C_CNN
    for i in range(C):  
        A, Z =  model.get_activatoins(X_chosen, i)
        display_comparaison_layer(A, Z)



def display_kernel(array_4d, type, stage, max_par_fig=16):
    if not isinstance(array_4d, np.ndarray) or array_4d.ndim != 4:
        raise ValueError("Entrée invalide : un array NumPy à 4 dimensions est requis (nb_kernels, nb_layers, height, width).")

    nb_kernels, nb_layers, h, w = array_4d.shape

    for kernel_idx in range(nb_kernels):
        total_layers = nb_layers

        for start in range(0, total_layers, max_par_fig):
            end = min(start + max_par_fig, total_layers)
            batch = array_4d[kernel_idx, start:end]

            n = batch.shape[0]
            cols = min(4, n)
            rows = (n + cols - 1) // cols

            fig = plt.figure(figsize=(cols * 4, rows * 3))
            fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche
            for i in range(n):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(batch[i])
                plt.title(f'{type} K{kernel_idx} L{start + i}')
                plt.axis('off')
                plt.colorbar()

            plt.suptitle(f'Stage {stage} | Kernel {kernel_idx} (Layers {start} à {end - 1})', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


"""
display_layer:
=========DESCRIPTION=========
Function that display the kernels & biais

=========INPUT=========
numpy.array     array_3d :      the activation matrice
string          type     :      string to inform if is the kernel matrice or biais matrice
string          stage    :      string to inform the stage of the in CNN      
=========OUTPUT=========
void
"""
def display_biais(array_3d, type, stage, max_par_fig=12):

    
    if not isinstance(array_3d, np.ndarray) or array_3d.ndim != 3:
        raise ValueError("Entrée invalide : un array NumPy à 3 dimensions est requis.")
    
    total = array_3d.shape[0]
    
    for start in range(0, total, max_par_fig):
        end = min(start + max_par_fig, total)
        batch = array_3d[start:end]

        n = batch.shape[0]
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig = plt.figure(figsize=(cols * 4, rows * 3))
        fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(batch[i])
            plt.title(f'{type} Couche {stage}: {start + i}')
            plt.axis('off')
            plt.colorbar()

        plt.suptitle(f'{type} - {stage} (couches {start} à {end - 1})', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Laisser de l’espace pour le suptitle
        plt.show()

"""
display_kernel_and_biais:
=========DESCRIPTION=========
Function that display the kernels & biais

=========INPUT=========
dict    parametres :    containt all the information for the pooling operation

=========OUTPUT=========
void
"""
def display_kernel_and_biais(X, y, model):

    def set_mode():
        while(1):
            print("\n0: Exit")
            print("1: Activation")
            print("2: Kernel")
            print("3: Biais")

            str_answer = input("Qu'est ce que vous voulez faire ?\n")
            try:
                int_answer = int(str_answer)
            except:
                print("Veuilliez repondre que par 1, 2 ou 3")
                continue
            if (int_answer == 0):
                print("Exit")
                exit(0)

            if (int_answer == 1):
                print("Vous voulez inspecter les activations")
                return(1)
            
            elif (int_answer == 2):
                print("Vous voulez inspecter les kernel")
                return(2)
            
            elif (int_answer == 3):
                print("Vous voulez inspecter les biais")
                return(3)
    
            else:
                print("Veuilliez repondre que par 1, 2 ou 3")

    mode = set_mode()
    if mode == 0:
        return
    
    if mode == 1:
        display_activation(X, y, model)
        return
    
    for i, block in enumerate(model.layers):
        
        if isinstance(block.dense, Convolution):

            K = block.dense.K
            b = block.dense.b

            if mode == 2:
                display_kernel(K, "Conv", i)

            elif mode == 3:
                display_biais(b, "Biais", i)

def display_first_picture(model, X_test, y_final):

    #Affichage des 15 premières images
    fig = plt.figure(figsize=(16,8))
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche
    for i in range(1,16):

        # Prédiction des probabilités avec softmax
        
        if X_test[i].ndim == 3:
            y_pred = model.forward_propagation(X_test[i][None, ...], False).flatten()
            X_display = X_test[i].transpose(1, 2, 0)

        elif X_test[i].ndim == 2:
            y_pred = model.forward_propagation(X_test[i][None, None, ...], False).flatten()
            X_display = X_test[i]

        pred = np.argmax(y_pred)
        porcent = np.max(y_pred)

        plt.subplot(4,5, i)
        plt.imshow(X_display)

        plt.title(f"Value:{y_final[i]} Predict:{pred}  ({np.round(porcent, 2)}%)")
        plt.tight_layout()
        plt.axis("off")
    plt.show()

def display_dataset(model, X_test, y_final, nb_test):

    print("")
    for i in range(nb_test):
        
        while(1):
            index = input(f"Please enter a number between 1 and {X_test.shape[0]}: ")

            # Check if input is empty or invalid
            if not index.strip():  
                print("❌ Please enter a valid number.")
                continue
            
            try:
                index = int(index)
            except ValueError:
                print("❌ Invalid input. Please enter an integer.")
                continue

            # Exit condition
            if index < 0:
                print("Exiting")
                exit(0)
            
            elif index < X_test.shape[0]:
                break

        # Prédiction des probabilités avec softmax
        if X_test[index].ndim == 3:
            y_pred = model.forward_propagation(X_test[index][None, ...], False).flatten()
            X_display = X_test[index].transpose(1, 2, 0)

        elif X_test[index].ndim == 2:
            y_pred = model.forward_propagation(X_test[index][None, None, ...], False).flatten()
            X_display = X_test[index]

        pred = np.argmax(y_pred)
        porcent = np.max(y_pred)

        # Création de la figure avec 2 sous-graphiques (image + histogramme)
        fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
        fig.canvas.mpl_connect('key_press_event', handle_key)  # Connecte l'événement clavier
        
        axs[0].imshow(X_display)
        axs[0].set_title(f"Value:{y_final[index]} Predict:{pred} ({np.round(porcent, 2)}%)")
        axs[0].axis("off")

        # Affichage de l'histogramme des probabilités
        axs[1].bar(range(len(y_pred)), y_pred, color="blue")
        axs[1].set_xticks(range(len(y_pred)))
        axs[1].set_xlabel("Classes")
        axs[1].set_ylabel("Probability")
        axs[1].set_ylim(0, 1)

        # Ajout des lignes horizontales tous les 0.1
        axs[1].set_yticks([i / 10 for i in range(11)])  # De 0.0 à 1.0 par pas de 0.1
        axs[1].grid(axis='y', linestyle='--', linewidth=0.5, color='red')  # Ligne fine et discrète

        plt.tight_layout()
        plt.show()