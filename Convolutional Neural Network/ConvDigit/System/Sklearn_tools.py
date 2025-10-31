
import numpy as np

def train_test_split(X, y, test_size, dataset_size):
    # Calculer la taille du test
    n_samples = dataset_size
    test_size = int(n_samples * test_size)  # Nombre d'exemples pour le test
    
    # Générer un masque de mélanger les indices
    indices = np.random.permutation(n_samples)
    
    # Séparer les indices pour le train et le test
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # Sélectionner les ensembles d'entraînement et de test
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
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