# Projet CNN from Scratch

Ce projet documente la création progressive d’un **réseau de neurones convolutifs (CNN)** implémenté entièrement à la main en Python, sans framework de deep learning.  
Chaque fichier correspond à une étape d’évolution du modèle, depuis les bases mathématiques jusqu’à une implémentation plus réaliste et optimisée.

---

## Structure du projet

### `convolution/`
Création des fonctions de base pour le CNN.

---

### `convolution02.py`
Création du premier modèle de CNN :
- Un seul kernel
- Une seule couche

---

### `convolution03.py`
Mise en place de l’abstraction par :
- Produit matriciel  
- Convolution matricialisée  
Objectif : transformer les convolutions en produits matriciels.

---

### `convolution04.py`
Essais avec une fonction d’activation de plus grande amplitude.

---

### `convolution05.py`
Ajouts majeurs :
- Variable **`dimension`** pour stocker les informations de construction du CNN  
- Variable **`parametre`** pour stocker :
  - Kernels  
  - Biais  
  - Fonctions d’activation  
- Variable **`activation`** pour stocker les activations au fur et à mesure  
- Création d’un CNN avec plusieurs couches

---

### `convolution06.py`
- Essais avec une activation plus grande  
- CNN avec davantage de fonctions

---

### `convolution07.py`
- Création de l’input et de la prédiction de manière aléatoire  
- Ajout de `tqdm` pour le suivi de progression

---

### `convolution08.py`
- Ajout de la dérivée de la fonction de coût  
- Calcul de l’accuracy  
- Affichage des données internes du CNN

---

### `convolution09.py`
- Ajout de listes et de `tuple_size`  
- Externalisation de l’affichage

---

### `convolution10.py`
- Ajout des dérivées :
  - Sigmoïde  
  - ReLU  
- Complétion de l’équation de la backpropagation  
- Ajout de la fonction de **pooling**  
- Ajout du **stride**  
- Découpage de la fonction `forward_propagation`

---

### `convolution11.py`
- Ajout de la fonction `show_information`

---

### `convolution12.py`
- Changement d’optimiseur :
  - Passage de la descente de gradient à **Adam**

---

### `convolution13.py`
- Identique à `convolution12.py`

---

### `convolution14.py`
Découpage de l’initialisation en plusieurs fonctions :
- `initialisation_extraction`  
- `initialisation_pooling`  
- `initialisation_kernel`

---

### `convolution15.py`
- Ajout du **padding**  
- Création de la fonction `adding_padding`  
- Ajout du mode **padding automatique**

---

### `convolution16.py`
- Ajout d’une dimension supplémentaire pour représenter l’existence :
(1, hauteur, largeur)


---

### `convolution17.py`
- Ajout d’une documentation complète des fonctions  
- Découpage de la fonction de backpropagation :
- `back_propagation_pooling`  
- `back_propagation_kernel`  
- Création de plusieurs kernels sur un même stage

---

### `convolution18.py`
- Chaque kernel multiplie chaque activation par son nombre de couches  
- Ajout de fonctions pour afficher les couches du CNN

---

### `convolution19.py`
- Documentation complète de toutes les fonctions  
- Ajout de nouvelles fonctions d’affichage des couches

---

### `convolution20.py`
- Optimisation des calculs (assistance ChatGPT)

---

### `convolution21.py`
- Création de kernels **3D**  
- Mise en place d’un **vrai CNN**

---

### `convolution22.py`
- Utilisation de **NumPy** et **SciPy**  
- Optimisation de la corrélation et de la convolution

---

### `convolution23.py`
- Tests pour vérifier si le modèle est correctement calibré

---

### `convolution24.py`
- Ajout de la fonction d’activation **Tanh**  
- Initialisation spécifique selon la fonction d’activation

---

### `convolution24b.py`
- Identique à `convolution24.py`  
- Input avec un **pattern en croix** pour calibrer le modèle

---

### `convolution25.py`
- Tentative de création d’un CNN avec une **vraie convolution**
- Sans déformation des inputs

---

## Objectif du projet

- Comprendre en profondeur le fonctionnement interne des CNN  
- Implémenter manuellement :
- Forward propagation  
- Backpropagation  
- Optimisation  
- Pooling, padding, stride  
- Explorer les performances et les optimisations bas niveau

---

## Auteur
Projet personnel à but pédagogique.
