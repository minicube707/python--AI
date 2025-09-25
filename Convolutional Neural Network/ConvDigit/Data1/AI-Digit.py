
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from datetime import datetime
from pathlib import Path

# Ajouter le dossier parent de Data1/ (donc C:/) à sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from System.Preprocessing import preprocessing, handle_key
from System.Mathematical_function import softmax
from System.File_Management import file_management, select_model, load_model, save_model
from System.Deep_Learning import convolution_neuron_network
from System.Initialisation_CNN import initialisation_AI, initialisation_affectation
from System.Propagation import forward_propagation
from System.Set_mode import set_mode
from System.Manage_logbook import fill_information, add_model

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)


#Data_Digit
with np.load("data/load_digits.npz") as f:
        X, y = f["data"], f["target"]

# Forme d'entrée (canaux, hauteur, largeur)
input_shape = (1, 8, 8)

# ============================
#     PRÉTRAITEMENT DONNÉES
# ============================
X_train, y_train, X_test, y_test, transformer = preprocessing(X, y, input_shape)

# ============================
#         PARAMÈTRES
# ============================

# Nombre d'itérations
nb_iteration = 0
max_attempts = 100
min_confidence_score = 0.25

# Mode d'exécution (1: train + save, 2: load + save, 3: load)
mode = set_mode()

# Paramètres d'apprentissage
# CNN
learning_rate_CNN = 0.0005
beta1 = 0.9
beta2 = 0.999

# DNN
learning_rate_DNN = 0.0001


print("\nInfo Training")
print("Nombre d'iteration: ", nb_iteration);
print("Max attempts: ", max_attempts)
print("Min confidence score: ", min_confidence_score)

print("\nInfo CNN")
print("Learning rate: ", learning_rate_CNN)
print("Beta1: ", beta1)
print("Beta2: ", beta2)

print("\nInfo DNN")
print("Learning rate: ", learning_rate_DNN)

if mode in {1}:

    # ============================
    #     INITIALISATION CNN
    # ============================

    # Structure CNN : (kernel_size, stride, padding, nb_kernels, type_layer, activation)
    dimensions_CNN = {  "1" :(3, 1, 0, 128, "kernel", "relu"),
                        "2" :(2, 2, 0, 1, "pooling", "max"), 
                        "3" :(2, 1, 0, 64, "kernel", "sigmoide")
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
    model, model_info = select_model(module_dir, "model_logbook.csv")
    parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN = load_model(module_dir, model)
    _, parametres_grad = initialisation_affectation(dimensions_CNN, tuple_size_activation)    



if mode in {1, 2}:
    # ============================
    #       TRAINNING
    # ============================

    # Entraînement d'un nouveau modèle
    parametres_CNN, parametres_DNN, test_accu, test_conf, test_loss, elapsed_time_minutes = convolution_neuron_network (
        X_train, y_train, X_test, y_test,
        nb_iteration,
        parametres_CNN, parametres_grad, parametres_DNN,
        dimensions_CNN,
        tuple_size_activation,
        learning_rate_CNN, beta1, beta2, learning_rate_DNN,
        max_attempts, min_confidence_score
    )

    # ============================
    #          SAVE
    # ============================

    # Sauvegarde du meilleur modèle entraîné ou chargé
    name_model = file_management(test_accu, test_conf)
    print(name_model)
    save_model(module_dir, name_model, (parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN))

    date = datetime.today()
    date = date.strftime('%d/%m/%Y')
    str_size = ','.join(str(v[0]) for v in dimensions_CNN.values() if v[4] == 'kernel')
    str_nb_kernel = ','.join(str(v[3]) for v in dimensions_CNN.values() if v[4] == 'kernel')

    if mode in {1}:
        nb_epoch = nb_iteration
        training_time = elapsed_time_minutes
        baseline_mode = "X"
        nb_fine_tunning = 0

    else:
        nb_epoch = float(model_info["nb_epoch"]) + nb_iteration
        training_time = float(model_info["training_time_(min)"]) + elapsed_time_minutes
        baseline_mode = model_info["name"]
        nb_fine_tunning = float(model_info["Number_fine_tunning"]) + 1

    new_log =  fill_information(name_model, date,
                                nb_epoch, max_attempts, min_confidence_score,
                                training_time, 
                                test_accu, test_conf, test_loss,
                                str_size, str_nb_kernel, 
                                learning_rate_CNN, learning_rate_DNN, beta1, beta2, 
                                len(y_train), len(y_test), 
                                baseline_mode, nb_fine_tunning)
    
    add_model(new_log, "LogBook", "model_logbook.csv")

#display_kernel_and_biais(parametres_CNN)

#______________________________________________________________#

C_CNN = len(dimensions_CNN.keys())
C_DNN = len(parametres_DNN) // 2

y_final = transformer.inverse_transform(y_test)

#Affichage des 15 premières images
fig = plt.figure(figsize=(16,8))
fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche
for i in range(1,16):

    # Prédiction des probabilités avec softmax
    _, activation_DNN = forward_propagation(X_test[i], parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN)
    probabilities = softmax(activation_DNN["A" + str(C_DNN)].T)
    pred = np.argmax(probabilities)
    porcent = np.max(probabilities)

    plt.subplot(4,5, i)
    plt.imshow(X_test[i].reshape(8, 8), cmap="gray")
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
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Connecte l'événement clavier
    
    # Affichage de l'image
    axs[0].imshow(X_test[index].reshape(8, 8), cmap="gray")
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
