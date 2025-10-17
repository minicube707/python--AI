
import numpy as np
import matplotlib.pyplot as plt

from .Preprocessing import handle_key

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
                plt.imshow(batch[i], cmap='gray')
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
            plt.imshow(batch[i], cmap='gray')
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
def display_kernel_and_biais(parametres):
    for key, value in parametres.items():
        if isinstance(value, np.ndarray):

            
            if key.startswith('K'):
                sqrt = np.int8(np.sqrt(value.shape[2]))
                K = value.reshape(value.shape[0], value.shape[1], sqrt, sqrt)
                display_kernel(K, "Kernel", key[-1])
        
            elif key.startswith('b'):
                sqrt = np.int8(np.sqrt(value.shape[1]))
                B = value.reshape(value.shape[0], sqrt, sqrt)
                display_biais(B, "Biais", key[-1])


"""
display_comparaison_layer:
=========DESCRIPTION=========
Function that display the kernels & biais

=========INPUT=========
numpy.array     y :             the target
numpy.array     y_pred :        the prediction of the model

=========OUTPUT=========
void
"""
def display_comparaison_layer(y, y_pred, max_par_fig=12):
    """
    Affiche chaque couche de deux tableaux 3D (y et y_pred) côte à côte,
    répartis sur plusieurs figures si nécessaire (max_par_fig par figure).
    """

    if y.shape != y_pred.shape or y.ndim != 3:
        raise ValueError("y et y_pred doivent être des arrays 3D de même forme (D, H, W)")

    total_couches = y.shape[0]

    for start in range(0, total_couches, max_par_fig):
        end = min(start + max_par_fig, total_couches)
        n = end - start

        cols = min(4, n)  # 4 paires par ligne
        rows = np.int8(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols, 3 * rows))
        fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche

        # Assurer que axes est 2D même pour une seule ligne
        if rows == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(n):
            layer_idx = start + i
            row = i // cols
            col = i % cols

            ax_y = axes[row, col * 2]
            ax_pred = axes[row, col * 2 + 1]

            ax_y.imshow(y[layer_idx], cmap='gray')
            ax_y.set_title(f'Y - Couche {layer_idx}')
            ax_y.axis('off')

            ax_pred.imshow(y_pred[layer_idx], cmap='gray')
            ax_pred.set_title(f'Prediction - Couche {layer_idx}')
            ax_pred.axis('off')
            ax_pred.colorbar()

        # Masquer les axes inutilisés
        total_axes = rows * cols * 2
        for j in range(n * 2, total_axes):
            row = j // (cols * 2)
            col = j % (cols * 2)
            axes[row, col].axis('off')

        plt.suptitle(f'Couches {start} à {end - 1}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()