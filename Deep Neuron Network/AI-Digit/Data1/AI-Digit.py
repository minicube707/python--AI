
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from Preprocessing import preprocessing
from Deep_Neuron_Network import deep_neural_network, foward_propagation, softmax
from File_Management import file_management, select_model, get_hidden_layers

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

load = False

#Data_Digit
with np.load("data/load_digits.npz") as f:
        X, y = f["data"], f["target"]

X = X.reshape(X.shape[0], -1)
y = y.reshape(y.shape[0], 1)


hidden_layer = (16,16)
learning_rate = 0.05
n_iteration = 5_000


X_train, y_train, X_test, y_test, transformer = preprocessing(X, y)

if load:    
    #Load the model
    model = select_model()
    with open("Model/" + str(model), 'rb') as file:
        parametres = pickle.load(file)
        hidden_layer = get_hidden_layers(model)

else:   
    parametres, dimensions, test_accu = deep_neural_network(X_train, y_train, X_test, y_test, hidden_layer, learning_rate, n_iteration)

if not load:

    #Save the best model
    name_model = file_management(hidden_layer, test_accu)
    print(name_model)
    with open("Model/" + name_model, 'wb') as file:
        pickle.dump(parametres, file)


#______________________________________________________________#
X_final = X_test.T
y_final = y_test.T
y_final = transformer.inverse_transform(y_final)

C = len(parametres) // 2
activation = foward_propagation(X_final.T, parametres)


#Affichage des 15 premières images
plt.figure(figsize=(16,8))
for i in range(1,16):
    # Prédiction des probabilités avec softmax
    print(softmax(activation["A" + str(C)].T))
    probabilities = softmax(activation["A" + str(C)].T)[i,:]
    pred = np.argmax(probabilities)
    porcent = np.max(probabilities)

    plt.subplot(4,5, i)
    plt.imshow(X_final.reshape((X_final.shape[0], 8, 8))[i], cmap="gray")
    plt.title(f"Value:{y_final[i]} Predict:{pred}  ({np.round(porcent, 2)}%)")
    plt.tight_layout()
    plt.axis("off")
plt.show()  

nb_test = 10
print("")
for i in range(nb_test):
    index = int(input(f"Please enter a number between 1 and {X_final.shape[0]}: "))
    if index == index < 0:
        break
    
    # Prédiction des probabilités avec softmax
    probabilities = softmax(activation["A" + str(C)].T)[index,:]
    pred = np.argmax(probabilities)
    porcent = np.max(probabilities)

    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})

    # Affichage de l'image
    axs[0].imshow(X_final.reshape((X_final.shape[0], 8, 8))[index], cmap="gray")
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
