
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

def model(X, W, B):
    Z = algebre(X, W, B)
    A = sigmoide(Z)
    return A


def init_animation(nb_iteraton):
    plt.ion()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Loss
    line_log, = axes[0].plot([], [], label="log_loss")
    line_dx,  = axes[1].plot([], [], label="dx_log")

    # Poids
    line_W, = axes[2].plot([], [], label="W")
    line_B, = axes[2].plot([], [], label="B")

    # Model
    line_error, = axes[3].plot([], [], color='orange', label="error")
    scatter_target = axes[3].scatter([], [], color='r', label="target")
    line_model, = axes[3].plot([], [], color='b', label="model")


    axes[0].set_title("Log loss")
    axes[1].set_title("Gradient loss")
    axes[2].set_title("Poids et biais")
    axes[3].set_title("Modèle")

    for i, ax in enumerate(axes):
        ax.legend()

        if i < 3:
            ax.set_xlim(0, nb_iteraton)
            ax.grid(True)

    axes[3].set_xlim(-2, 2)
    axes[3].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.show()

    return fig, axes, line_log, line_dx, line_W, line_B, line_model, scatter_target, line_error


def update_graph(
    axes,
    line_log, line_dx,
    line_W, line_B,
    line_model, scatter_target, line_error,
    log, dx_log,
    W_log, B_log,
    X_vis, X, y, W, B,
    max_iter
):
    # Loss
    line_log.set_data(range(len(log)), log)
    line_dx.set_data(range(len(dx_log)), dx_log)

    for i, ax in enumerate(axes):
        if i < 3:
            ax.set_xlim(0, max_iter)
            ax.grid(True)

    axes[0].relim()
    axes[0].autoscale_view()

    axes[1].relim()
    axes[1].autoscale_view()

    # Poids
    line_W.set_data(range(len(W_log)), W_log)
    line_B.set_data(range(len(B_log)), B_log)

    axes[2].relim()
    axes[2].autoscale_view()

    # Model update
    Y_vis = model(X_vis, W, B)
    Y_pred = float(model(X, W, B))

    line_model.set_data(X_vis, Y_vis)
    scatter_target.set_offsets([[X, y]])
    line_error.set_data([X, X], [Y_pred, y])

    axes[3].relim()
    axes[3].autoscale_view()



# ========
#   MAIN
# ========

learning_rate = 0.1
nb_iteraton = 3000

W = np.random.rand(1) * 2 - 1
B = np.random.rand(1) * 2 - 1

log = []
dx_log = []

# Avant la boucle principale, initialise les listes d'historique
W_log, B_log = [], []

#Graphique Initilisation
fig, axes, line_log, line_dx, line_W, line_B, line_model, scatter_target, line_error = init_animation(nb_iteraton)
X_vis = np.linspace(-2, 2, 100)


#Premier PASSAGE
X = 1
y = 1

print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print("W: ", W)
print("B: ", B)



for i in tqdm(range(nb_iteraton)):
    
    #Foreward propagation
    Z = algebre(X, W, B)
    A = sigmoide(Z)

    log.append(log_loss(A, y))
    dx_log.append(dx_log_loss(y, A))

    W_log.append(W.copy())
    B_log.append(B.copy())

    #Backpropagation
    dZ = A - y          #dL/dZ 
    dW = X * dZ         #dL/dW
    db = dZ             #dL/dB
    W -= dW * learning_rate
    B -= db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target, line_error,
        log, dx_log,
        W_log, B_log,
        X_vis, X, y, W, B,
        nb_iteraton
    )

    if i % 10 == 0:
        plt.pause(0.001)


print("")
print("W: ", W)
print("B: ", B)

print("Loss final ", log_loss(A, y))
print("ACTIVATION final", sigmoide(algebre(X, W, B)))


#DEUXIEME PASSAGE
y = 0

print("")
print("DEUXIEME apprentissage")
print("X: ", X)
print("y: ", y)
print("W: ", W)
print("B: ", B)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for i in tqdm(range(nb_iteraton)):
    
    #Foreward propagation
    Z = algebre(X, W, B)
    A = sigmoide(Z)

    log.append(log_loss(A, y))
    dx_log.append(dx_log_loss(y, A))

    W_log.append(W.copy())
    B_log.append(B.copy())

    #Backpropagation
    dZ = A - y          #dL/dZ 
    dW = X * dZ         #dL/dW
    db = dZ             #dL/dB
    W -= dW * learning_rate
    B -= db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,  line_error,
        log, dx_log,
        W_log, B_log,
        X_vis, X, y, W, B,
        nb_iteraton * 2
    )

    if i % 10 == 0:
        plt.pause(0.001)

print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log_loss(A, y))
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))


#Troisieme PASSAGE
X = -1
y = 0

print("")
print("Troisieme apprentissage")
print("X: ", X)
print("y: ", y)
print("W: ", W)
print("B: ", B)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for i in tqdm(range(nb_iteraton)):
    
    #Foreward propagation
    Z = algebre(X, W, B)
    A = sigmoide(Z)

    log.append(log_loss(A, y))
    dx_log.append(dx_log_loss(y, A))

    W_log.append(W.copy())
    B_log.append(B.copy())

    #Backpropagation
    dZ = A - y          #dL/dZ 
    dW = X * dZ         #dL/dW
    db = dZ             #dL/dB
    W -= dW * learning_rate
    B -= db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,
        log, dx_log,
        W_log, B_log,
        X_vis, X, y, W, B,
        nb_iteraton * 3
    )

    if i % 10 == 0:
        plt.pause(0.001)

print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log_loss(A, y))
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))


#Quatrieme PASSAGE
X = -1
y = 1

print("")
print("Quatrieme apprentissage")
print("X: ", X)
print("y: ", y)
print("W: ", W)
print("B: ", B)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for i in tqdm(range(nb_iteraton)):
    
    #Foreward propagation
    Z = algebre(X, W, B)
    A = sigmoide(Z)

    log.append(log_loss(A, y))
    dx_log.append(dx_log_loss(y, A))

    W_log.append(W.copy())
    B_log.append(B.copy())

    #Backpropagation
    dZ = A - y          #dL/dZ 
    dW = X * dZ         #dL/dW
    db = dZ             #dL/dB
    W -= dW * learning_rate
    B -= db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,
        log, dx_log,
        W_log, B_log,
        X_vis, X, y, W, B,
        nb_iteraton * 4
    )

    if i % 10 == 0:
        plt.pause(0.001)

print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log_loss(A, y))
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))

#Cinquieme PASSAGE
X = 0
y = 0.5

print("")
print("Cinquieme apprentissage")
print("X: ", X)
print("y: ", y)
print("W: ", W)
print("B: ", B)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for i in tqdm(range(nb_iteraton)):
    
    #Foreward propagation
    Z = algebre(X, W, B)
    A = sigmoide(Z)

    log.append(log_loss(A, y))
    dx_log.append(dx_log_loss(y, A))

    W_log.append(W.copy())
    B_log.append(B.copy())

    #Backpropagation
    dZ = A - y          #dL/dZ 
    dW = X * dZ         #dL/dW
    db = dZ             #dL/dB
    W -= dW * learning_rate
    B -= db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,
        log, dx_log,
        W_log, B_log,
        X_vis, X, y, W, B,
        nb_iteraton * 5
    )

    if i % 10 == 0:
        plt.pause(0.001)

print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log_loss(A, y))
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))


#Sixieme PASSAGE
X = 1
y = 0.5

print("")
print("Sixieme apprentissage")
print("X: ", X)
print("y: ", y)
print("W: ", W)
print("B: ", B)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for i in tqdm(range(nb_iteraton)):
    
    #Foreward propagation
    Z = algebre(X, W, B)
    A = sigmoide(Z)

    log.append(log_loss(A, y))
    dx_log.append(dx_log_loss(y, A))

    W_log.append(W.copy())
    B_log.append(B.copy())

    #Backpropagation
    dZ = A - y          #dL/dZ 
    dW = X * dZ         #dL/dW
    db = dZ             #dL/dB
    W -= dW * learning_rate
    B -= db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,
        log, dx_log,
        W_log, B_log,
        X_vis, X, y, W, B,
        nb_iteraton * 6
    )

    if i % 10 == 0:
        plt.pause(0.001)

print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log_loss(A, y))
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))


#Septieme PASSAGE
X = -1
y = 0.5

print("")
print("Septieme apprentissage")
print("X: ", X)
print("y: ", y)
print("W: ", W)
print("B: ", B)

Z = algebre(X, W, B)
A = sigmoide(Z)

print("Loss", log_loss(A, y))
print("ACTIVATION", A)
print("")

for i in tqdm(range(nb_iteraton)):
    
    #Foreward propagation
    Z = algebre(X, W, B)
    A = sigmoide(Z)

    log.append(log_loss(A, y))
    dx_log.append(dx_log_loss(y, A))

    W_log.append(W.copy())
    B_log.append(B.copy())

    #Backpropagation
    dZ = A - y          #dL/dZ 
    dW = X * dZ         #dL/dW
    db = dZ             #dL/dB
    W -= dW * learning_rate
    B -= db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,
        log, dx_log,
        W_log, B_log,
        X_vis, X, y, W, B,
        nb_iteraton * 7
    )

    if i % 10 == 0:
        plt.pause(0.001)

print("")
print("W: ", W)
print("B: ", B)
print("Loss final ", log_loss(A, y))
print("y: ", y)
print("ACTIVATION final", sigmoide(algebre(X, W, B)))