# Project DNN

Ce projet implémente un **réseau de neurones profonds (DNN)** pour [objectif du projet : classification, détection, etc.].  

---

## Versions

### Prototype  
Le dossier `deep_neuron_network` contient environ 13 fichiers de démonstration montrant comment construire un DNN **from scratch**.  
Pour plus de détails sur les fichiers, consultez le README du dossier.

### Version finale  
Le dossier `AI-Digit (New Model)` contient :  
- Le modèle final  
- Un code propre, optimisé et commenté  

**C'est la version recommandée pour l'exécution.**

---

## Dépendances

Ce projet utilise uniquement des bibliothèques Python **installables via pip**.  
Pour installer toutes les dépendances, exécute :

```bash
pip install numpy tqdm matplotlib scikit-learn pandas seaborn scipy pygame
```

---
Pour gérer les dépendances avec **UV**, procédez comme suit :

1. Installer UV si ce n’est pas déjà fait :

```bash
pip install uv
```

2. Ensuite initialiser **uv** dans dossier :
```bash
uv init
```

3. Ajouter les dépendances du projet :
```bash
uv add numpy tqdm matplotlib scikit-learn pandas seaborn scipy pygame
```