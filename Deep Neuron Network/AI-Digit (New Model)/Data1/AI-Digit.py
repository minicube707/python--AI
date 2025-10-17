import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from datetime import datetime
from pathlib import Path

# Ajouter le dossier parent de Data1/ (donc C:/) à sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from System.Preprocessing import preprocessing, handle_key, show_information_setting
from System.Mathematical_function import softmax
from System.File_Management import file_management, select_model, load_model, save_model
from System.Set_mode import set_mode
from System.Manage_logbook import fill_information, add_model, show_all_info_model
from System.Deep_Neuron_Network import show_information_DNN, initialisation_DNN, foward_propagation_DNN
from Deep_Neuron_Network import deep_neuron_network

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
nb_iteration = 100
max_attempts = 1
min_confidence_score = 0
validation_size = 50

# Paramètres d'apprentissage

# DNN
learning_rate_DNN = 0.001
alpha = 0.001

if (validation_size > len(y_test)):
    print("Validation set two large")
    exit(1)

show_information_setting(nb_iteration, max_attempts, min_confidence_score,  alpha, learning_rate_DNN, validation_size)

# Mode d'exécution (1: train + save, 2: load + save, 3: load)
mode = set_mode()

if mode in {1}:

    # Structure DNN : (number of neurone, activations) 
    dimensions_DNN = {
        "1": (64, "relu"),
        "2": (64, "relu"),
        "3": (0,  "relu")
    }

    # Mode de padding : 'auto' = calcul automatique
    padding_mode = "auto"

    #Initialisation
    parametres_DNN = initialisation_DNN(dimensions_DNN, 64, y_train.shape[1])

else:
    # ============================
    #       SELECT A MODEL
    # ============================

    # Chargement du modele existant
    model, model_info = select_model(module_dir, "model_logbook.csv")
    parametres_DNN, dimensions_DNN = load_model(module_dir, model)  


show_information_DNN(parametres_DNN, dimensions_DNN)


if mode in {1, 2}:
    # ============================
    #       TRAINNING
    # ============================

    # Entraînement d'un nouveau modèle
    parametres_DNN, test_accu, test_conf, test_loss, elapsed_time_minutes = deep_neuron_network (
        X_train, y_train, X_test, y_test,
        nb_iteration,parametres_DNN, dimensions_DNN,
        alpha, learning_rate_DNN,
        max_attempts, min_confidence_score, validation_size
    )

    # ============================
    #          SAVE
    # ============================

    # Sauvegarde du meilleur modèle entraîné ou chargé
    name_model = file_management(test_accu, test_conf)
    print(name_model)
    save_model(module_dir, name_model, (parametres_DNN, dimensions_DNN))

    date = datetime.today()
    date = date.strftime('%d/%m/%Y')
 
    if mode in {1}:
        nb_epoch = nb_iteration
        training_time = elapsed_time_minutes
        baseline_mode = "X"
        nb_fine_tunning = 0

        #DNN
        str_size_DNN = ','.join(str(v[0]) for v in dimensions_DNN.values())
        str_function_DNN = ','.join(str(v[1]) for v in dimensions_DNN.values())

    else:
        nb_epoch = float(model_info["nb_epoch"]) + nb_iteration
        training_time = float(model_info["training_time_(min)"]) + elapsed_time_minutes
        baseline_mode = model_info["name"]
        nb_fine_tunning = float(model_info["Number_fine_tunning"]) + 1

        #DNN
        str_size_DNN = model_info["neurons_number"]
        str_function_DNN = model_info["activation_function_DNN"]

    new_log =  fill_information(name_model, date, training_time,
                    nb_epoch,  max_attempts, min_confidence_score, alpha,
                    test_loss, test_accu, test_conf, 
                    learning_rate_DNN, str_size_DNN, str_function_DNN,
                    len(y_train), len(y_test), 
                    baseline_mode, nb_fine_tunning, validation_size)
    
    add_model(new_log, "LogBook", "model_logbook.csv")

#______________________________________________________________#

C_DNN = len(parametres_DNN) // 2

y_final = transformer.inverse_transform(y_test)

#Affichage des 15 premières images
fig = plt.figure(figsize=(16,8))
fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche
for i in range(1,16):

    # Prédiction des probabilités avec softmax
    activation_DNN = foward_propagation_DNN(X_test[i], parametres_DNN, dimensions_DNN, C_DNN, alpha)
    probabilities = softmax(activation_DNN["A" + str(C_DNN)])
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
    activation_DNN = foward_propagation_DNN(X_test[index], parametres_DNN, dimensions_DNN, C_DNN, alpha)
    probabilities = softmax(activation_DNN["A" + str(C_DNN)]).flatten()
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

    # Ajout des lignes horizontales tous les 0.1
    axs[1].set_yticks([i / 10 for i in range(11)])  # De 0.0 à 1.0 par pas de 0.1
    axs[1].grid(axis='y', linestyle='--', linewidth=0.5, color='red')  # Ligne fine et discrète

    plt.tight_layout()
    plt.show()
