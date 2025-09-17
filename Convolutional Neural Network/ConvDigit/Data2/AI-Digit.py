
import numpy as np
import matplotlib.pyplot as plt
import os

from Preprocessing import preprocessing
from Mathematical_function import softmax
from File_Management import file_management, select_model, load_model, save_model
from Deep_Learning import convolution_neuron_network
from Initialisation_CNN import initialisation_AI, initialisation_affectation
from Propagation import forward_propagation
from Set_mode import set_mode

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)


#Data_Digit
with np.load("data/mnist.npz") as f:
        X, y = f["x_train"], f["y_train"]

# Forme d'entrée (canaux, hauteur, largeur)
input_shape = (1, 28, 28)

# ============================
#     PRÉTRAITEMENT DONNÉES
# ============================
X_train, y_train, X_test, y_test, transformer = preprocessing(
    X[:2000], y[:2000], input_shape
)

# ============================
#         PARAMÈTRES
# ============================

# Nombre d'itérations
nb_iteration = 1

# Mode d'exécution (1: train + save, 2: load + save, 3: load)
mode = set_mode()

# Paramètres d'apprentissage
# CNN
learning_rate_CNN = 0.005
beta1 = 0.9
beta2 = 0.999

# DNN
learning_rate_DNN = 0.001

if mode in {1}:

    # ============================
    #     INITIALISATION CNN
    # ============================

    # Structure CNN : (kernel_size, stride, padding, nb_kernels, type_layer, activation)
    dimensions_CNN = {
        "1": (3, 1, 0, 32, "kernel", "relu"),
        "2": (2, 2, 0, 1, "pooling", "max"),
        "3": (3, 1, 0, 64, "kernel", "relu"),
        "4": (2, 2, 0, 1, "pooling", "max"),
        "5": (3, 1, 0, 64, "kernel", "sigmoide")
    }

    # Structure DNN : (hidden layer) 
    hidden_layer = (64, 64)

    # Mode de padding : 'auto' = calcul automatique
    padding_mode = "auto"

    #Initialisation
    parametres_CNN, parametres_grad, parametres_DNN, dimensions_CNN, tuple_size_activation = initialisation_AI (
        input_shape, dimensions_CNN, padding_mode, hidden_layer, y_train.shape
    )


else:
    # ============================
    #       SELECT A MODEL
    # ============================

    # Chargement du modele existant
    model = select_model()
    parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN = load_model(model)
    _, parametres_grad = initialisation_affectation(dimensions_CNN, tuple_size_activation)    



if mode in {1, 2}:
    # ============================
    #       TRAINNING
    # ============================

    # Entraînement d'un nouveau modèle
    parametres_CNN, parametres_DNN, dimensions_CNN, test_accu, test_conf = convolution_neuron_network (
        X_train, y_train, X_test, y_test,
        nb_iteration,
        parametres_CNN, parametres_grad, parametres_DNN,
        dimensions_CNN,
        tuple_size_activation,
        learning_rate_CNN, beta1, beta2, learning_rate_DNN
    )

    # ============================
    #          SAVE
    # ============================

    # Sauvegarde du meilleur modèle entraîné ou chargé
    name_model = file_management(test_accu, test_conf, dimensions_CNN)
    print(name_model)
    save_model(name_model, (parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN))


#display_kernel_and_biais(parametres_CNN)

#______________________________________________________________#

C_CNN = len(dimensions_CNN.keys())
C_DNN = len(parametres_DNN) // 2

y_final = transformer.inverse_transform(y_test)

#Affichage des 15 premières images
plt.figure(figsize=(16,8))
for i in range(1,16):

    # Prédiction des probabilités avec softmax
    _, activation_DNN = forward_propagation(X_test[i], parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN)
    probabilities = softmax(activation_DNN["A" + str(C_DNN)].T)
    pred = np.argmax(probabilities)
    porcent = np.max(probabilities)

    plt.subplot(4,5, i)
    plt.imshow(X_test[i].reshape(input_shape[1], input_shape[2]), cmap="gray")
    plt.title(f"Value:{y_final[i]} Predict:{pred}  ({np.round(porcent, 2)}%)")
    plt.tight_layout()
    plt.axis("off")
plt.show()  

nb_test = 10
print("")
for i in range(nb_test):
    index = int(input(f"Please enter a number between 1 and {X_test.shape[0]}: "))
    if index == index < 0:
        break
    
    # Prédiction des probabilités avec softmax
    _, activation_DNN = forward_propagation(X_test[index], parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN)
    probabilities = softmax(activation_DNN["A" + str(C_DNN)].T).flatten()
    pred = np.argmax(probabilities)
    porcent = np.max(probabilities)

    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})

    # Affichage de l'image
    axs[0].imshow(X_test[index].reshape(input_shape[1], input_shape[2]), cmap="gray")
    axs[0].set_title(f"Value:{y_final[index]} Predict:{pred} ({np.round(porcent, 2)}%)")
    axs[0].axis("off")

    # Affichage de l'histogramme des probabilités
    axs[1].bar(range(len(probabilities)), probabilities, color="blue")
    axs[1].set_xticks(range(len(probabilities)))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
