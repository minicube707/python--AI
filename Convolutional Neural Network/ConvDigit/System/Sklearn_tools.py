
import numpy as np

def train_test_split(X, y, test_size, dataset_size):

    # Taille totale du dataset d'origine
    total_size = X.shape[0]

    # Si dataset_size n'est pas précisé, on prend tout
    if dataset_size is None or dataset_size > total_size:
        dataset_size = total_size
    
    # Mélange initial des indices pour choisir un sous-ensemble du dataset
    all_indices = np.random.permutation(total_size)[:dataset_size]
    
    # Sous-échantillonnage de X et y
    X_subset = X[all_indices]
    y_subset = y[all_indices]
    
    # Calcul du nombre d'exemples pour le test
    test_count = int(dataset_size * test_size)
    
    # Mélange à nouveau pour séparer train/test
    indices = np.random.permutation(dataset_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # Création des ensembles
    X_train, X_test = X_subset[train_indices], X_subset[test_indices]
    y_train, y_test = y_subset[train_indices], y_subset[test_indices]
    
    return X_train, X_test, y_train, y_test

def Label_binarizer(y):
    # Trouver les classes uniques dans y
    classes = np.unique(y)
    
    # Créer une matrice de zéros de forme (n_samples, n_classes)
    one_hot = np.zeros((y.size, classes.size))
    
    # Remplir la matrice avec des 1 aux positions appropriées
    for i, label in enumerate(y):
        one_hot[i, np.where(classes == label)[0]] = 1
    
    return np.int8(one_hot)