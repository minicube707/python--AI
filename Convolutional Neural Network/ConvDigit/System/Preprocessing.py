
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import IsolationForest

from .Sklearn_tools import train_test_split, Label_binarizer

def get_data_shape(X, y):

    if y.ndim == 1:
        n_labels = len(np.unique(y))
    else:
        n_labels = y.shape[1]

    # Forme d'entrée (nb_data, hauteur, largeur)
    if X.ndim == 3:
        input_shape = (1, X.shape[1], X.shape[2])
        output_shape = n_labels

        print("")
        print("X has 2 dimensions")
        print("Input shape: ", input_shape)
        print("Output shape: ", output_shape)

    # Forme d'entrée (nb_data, hauteur, largeur, cannaux)
    elif X.ndim == 4:
        _, _, _, channel = X.shape
        input_shape = (channel, X.shape[1], X.shape[2])
        output_shape = n_labels

        print("")
        print("X has 3 dimensions")
        print("Input shape: ", input_shape)
        print("Output shape: ", output_shape)

    else:
        print(f"Error: Data with wrong shape X:({X.shape})")
        exit(1)
        
    return input_shape, output_shape


def handle_key(event):
    if event.key == ' ':
        plt.close(event.canvas.figure)  # Ferme la fenêtre associée


def preprocessing(X, y, hyperparams, dataset):
    
    print("")
    print("Data shape")
    print("X:",X.shape)
    print("Y:",y.shape)

    contamination = hyperparams.contamination
    
    
    """
    Affiche les 15 premières images de chaque classe du dataset.
    """
    classes = np.unique(y)
    for cls in classes:
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(f"Classe {cls}", fontsize=16)
        fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche
        
        # Récupère les indices des images correspondant à la classe cls
        indices = np.where(y == cls)[0][:15]  # 15 premières images
        for i, idx in enumerate(indices):
            plt.subplot(3, 5, i + 1)  # 3 lignes, 5 colonnes
            plt.imshow(X[idx])

            plt.title(f"{y[idx]}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    #______________________________________________________________#
    #Remove the bad data
    X_dim2 = X.reshape(X.shape[0], -1)
    model=IsolationForest(contamination=contamination)
    model.fit(X_dim2)
    outlier = model.predict(X_dim2) == 1
    X = X[outlier]
    y = y[outlier]


    #______________________________________________________________#
    #Split the dataset for the training
    X_train, X_test, y_train, y_test = train_test_split(X, y, dataset.ratio_test, dataset.dataset_size)
    
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))


    #______________________________________________________________#
    #Encode the labels for the trainning
    transformer=LabelBinarizer()
    transformer.fit(y_train)
    y_train = transformer.transform(y_train.reshape((-1, 1)))
    y_test = transformer.transform(y_test.reshape((-1, 1)))

    print("\nTrain")
    print("La dimension de X_train",X_train.shape)
    print("La dimension de y_train",y_train.shape)
    print(np.unique(y_train, return_counts=True))

    print("\nTest")
    print("La dimension de X_test",X_test.shape)
    print("La dimension de y_test",y_test.shape)
    print(np.unique(y_test, return_counts=True))
  
    New_X_train = X_train / X_train.max()
    New_X_test = X_test / X_train.max()

    

    #Pour les X se sont les variables en premier (ici les pixels) puis le nombres d'échantillons 
    #Pour les y se sont les labels d'abord puis le nombre d'échantillons
    print("\nNew_X_train.shape:", New_X_train.shape)
    print("New_X_test.shape:", New_X_test.shape)
    print("y_test.shape:", y_test.shape)
    print("y_train.shape:", y_train.shape)

    #Affichage des 15 premières images du dataset
    n = min(16, len(y_train))
    fig = plt.figure(figsize=(n,8))
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de touches 
    fig.suptitle("Train Dataset")
    for i in range(1,n):
        plt.subplot(4,5, i)
        plt.imshow(New_X_train[i], cmap="gray")

        plt.title(str(np.argmax(y_train[i])))
        plt.axis("off")
    plt.tight_layout()    
    plt.show() 

    #Affichage des 15 premières images
    n = min(16, len(y_test))
    fig = plt.figure(figsize=(n,8))
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de touches 
    fig.suptitle("Test Dataset")
    for i in range(1,n):
        plt.subplot(4,5, i)
        plt.imshow(New_X_test[i], cmap="gray")
        

        plt.title(str(np.argmax(y_test[i])))
        plt.axis("off")
    plt.tight_layout()    
    plt.show() 

    return New_X_train, y_train, New_X_test, y_test, transformer