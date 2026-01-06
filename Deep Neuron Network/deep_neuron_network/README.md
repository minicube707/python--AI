
# Comment créer un Deep Neural Network (DNN) from Scratch

Ce dossier présente étape par étape **la construction d’un DNN from scratch** en Python. Chaque fichier montre une fonctionnalité ou une technique spécifique.

---

## Lancer le code

- Avec **pip** :  
```bash
python neuron_network01.py
```

- Avec **UV**:  
```bash
uv run neuron_network01.py
```

## Animation
Il y a deux fichiers qui contiennent des animations: 
- neuron_network01_animation.py
- neuron_network02_animation.py  

Ces scripts montrent comment un modèle apprend avec un ou deux points de données.

# Classique 
## 1. neuron_network01.py
- Modèle avec un seul neurone, une entrée et une sortie utilisant la fonction **Sigmoïde**.
- Variantes :  
  - `neuron_network01b.py` : le modèle apprend quelque chose, puis désapprend pour réapprendre autre chose.  
  - `neuron_network01bb.py` : montre que le sens du gradient est une convention ; le modèle peut apprendre même si le gradient est inversé.  

## 2. neuron_network02.py
- Modèle avec un neurone, une entrée, une sortie et **deux variables**.  

## 3. neuron_network03.py
- Modèle avec **deux neurones en série**, une entrée, une sortie et deux variables.  
- Variante : `neuron_network03b.py` : réinitialisation des paramètres au deuxième passage.

## 4. neuron_network04.py
- Modèle avec **trois neurones**, deux entrées et une sortie, deux variables, fonction Sigmoïde.

## 5. neuron_network05.py
- Modèle avec **cinq neurones**, deux entrées, **deux neurones cachés**, une sortie, fonction Sigmoïde.

## 6. neuron_network06.py
- Comme neuron_network05.py mais **activation LeakyReLU** pour la couche cachée.  
- Variantes :  
  - `neuron_network06b.py` : première couche aussi avec LeakyReLU.  
  - `neuron_network06bb.py` : fonction Tanh pour la couche cachée.

## 7. neuron_network07.py
- Modèle avec **six neurones**, deux entrées et quatre sorties.  
- Classification **multi-classe** avec quatre variables/features.

## 8. neuron_network08.py
- Introduction aux **matrices**, trois neurones, deux entrées, une sortie et deux variables (comme neuron_network04.py).  
- Variante : `neuron_network08b.py` : version corrigée.

## 9. neuron_network09.py
- Matrices avancées : modèle avec **3 couches**, 10 neurones, 4 entrées, 4 neurones cachés, 2 sorties, Sigmoïde.

## 10. neuron_network10.py
- Modularité : choisir **nombre de neurones et de couches**. Fonctionne avec Sigmoïde.

## 11. neuron_network11.py
- Vectorisation avec **NumPy** : le modèle reçoit tout le dataset, plus d’apprentissage variable par variable.

## 12. neuron_network12.py
- Modularité avancée : comme neuron_network10.py mais on peut aussi choisir **la fonction d’activation** (Sigmoïde, Tanh, LeakyReLU).

## 13. neuron_network13.py
- Modèle modulaire pour **classer les nombres de 0 à 15** en binaire vers leur base décimale.