
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  - y * np.log(A + epsilon) - (1-y) * np.log(1-A + epsilon)

def dx_log_loss(y_true, y_pred):
    return - y_true/y_pred + (1 - y_true)/(1 - y_pred)

def algebre(x, a, b):
    return a * x + b

def sigmoide(X):
    return 1/(1 + np.exp(-X))

#INITIALISATION
X = np.array([0, 1])
y = np.array([0, 1])

learning_rate = 0.001
nb_iteraton = 30_000

W = np.random.rand(1) * 2 - 1
B = np.random.rand(1) * 2 - 1

log = []
dx_log = []

# Avant la boucle principale, initialise les listes d'historique
W_log, B_log = [], []

#PREMIER PASSAGE
print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print("W: ", W)
print("B: ", B)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for _ in tqdm(range(nb_iteraton)):
    
    sum_log = 0
    sum_dx_log = 0

    sum_dW = 0
    sum_db = 0

    for j in range(X.size):

        #Foreward propagation
        Z = algebre(X[j], W, B)
        A = sigmoide(Z)

        sum_log += log_loss(A, y[j])
        sum_dx_log += dx_log_loss(y[j], A)
        
        #Backpropagation
        dZ = A - y[j]          #dL/dZ
        sum_dW += X[j] * dZ     #dL/dW
        sum_db += dZ            #dL/dB

    log.append(sum_log)
    dx_log.append(sum_dx_log)

    W_log.append(W.copy())
    B_log.append(B.copy())
    
    W -= sum_dW * learning_rate
    B -= sum_db * learning_rate


print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log_loss(A, y))
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
axes[0].plot(log)
axes[0].set_title("log")
axes[1].plot(dx_log)
axes[1].set_title("dx_log")
plt.tight_layout()
plt.show()

# Créer une figure
plt.figure(figsize=(12, 8))
plt.plot(W_log, label='W')
plt.plot(B_log, label='B')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()


#DEUXIEME PASSAGE
y = np.array([1, 0])

print("")
print("Deuxieme apprentissage")
print("X: ", X)
print("y: ", y)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for _ in tqdm(range(nb_iteraton)):
    
    sum_log = 0
    sum_dx_log = 0

    sum_dW = 0
    sum_db = 0

    for j in range(X.size):

        #Foreward propagation
        Z = algebre(X[j], W, B)
        A = sigmoide(Z)

        sum_log += log_loss(A, y[j])
        sum_dx_log += dx_log_loss(y[j], A)
        
        #Backpropagation
        dZ = A - y[j]          #dL/dZ
        sum_dW += X[j] * dZ     #dL/dW
        sum_db += dZ            #dL/dB

    log.append(sum_log)
    dx_log.append(sum_dx_log)

    W_log.append(W.copy())
    B_log.append(B.copy())
    
    W -= sum_dW * learning_rate
    B -= sum_db * learning_rate


print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log_loss(A, y))
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))

# Créer une figure avec deux sous-graphes côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 ligne, 2 colonnes
axes[0].plot(log)
axes[0].set_title("log")
axes[1].plot(dx_log)
axes[1].set_title("dx_log")
plt.tight_layout()
plt.show()

# Créer une figure
plt.figure(figsize=(12, 8))
plt.plot(W_log, label='W')
plt.plot(B_log, label='B')
plt.legend()
plt.title("Évolution des poids et biais")
plt.xlabel("Itérations")
plt.ylabel("Valeur")
plt.grid(True)
plt.tight_layout()
plt.show()