
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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
    scatter_target = axes[3].scatter([], [], color='r', label="target")
    line_model, = axes[3].plot([], [], color='b', label="model")
    line_error = LineCollection([], colors='orange', label="error")
    axes[3].add_collection(line_error)

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
    Y_pred = model(X, W, B)
    
    line_model.set_data(X_vis, Y_vis)
    scatter_target.set_offsets(np.c_[X, y])

    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    Y_pred = np.asarray(Y_pred).reshape(-1)

    segments = [
        [(X[i], Y_pred[i]), (X[i], y[i])]
        for i in range(len(X))
    ]
    line_error.set_segments(segments)

    axes[3].relim()
    axes[3].autoscale_view()



# ========
#   MAIN
# ========

learning_rate = 0.1
nb_iteraton = 2000

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
X = np.array([-1, 1])
y = np.array([0, 1])

print("")
print("Premier apprentissage")
print("X: ", X)
print("y: ", y)
print("W: ", W)
print("B: ", B)

for i in tqdm(range(nb_iteraton)):
    
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
        dZ = A - y[j]           #dL/dZ
        sum_dW += X[j] * dZ     #dL/dW
        sum_db += dZ            #dL/dB

    log.append(sum_log)
    dx_log.append(sum_dx_log)

    W_log.append(W.copy())
    B_log.append(B.copy())
    
    W -= sum_dW * learning_rate
    B -= sum_db * learning_rate

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
y = np.array([1, 0])

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
        dZ = A - y[j]           #dL/dZ
        sum_dW += X[j] * dZ     #dL/dW
        sum_db += dZ            #dL/dB

    log.append(sum_log)
    dx_log.append(sum_dx_log)

    W_log.append(W.copy())
    B_log.append(B.copy())
    
    W -= sum_dW * learning_rate
    B -= sum_db * learning_rate
    
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
X = np.array([-1, 1])
y = np.array([0.5, 0.5])

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
        dZ = A - y[j]           #dL/dZ
        sum_dW += X[j] * dZ     #dL/dW
        sum_db += dZ            #dL/dB

    log.append(sum_log)
    dx_log.append(sum_dx_log)

    W_log.append(W.copy())
    B_log.append(B.copy())
    
    W -= sum_dW * learning_rate
    B -= sum_db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,  line_error,
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
X = np.array([0.5, 1.5])
y = np.array([0, 1])

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
        dZ = A - y[j]           #dL/dZ
        sum_dW += X[j] * dZ     #dL/dW
        sum_db += dZ            #dL/dB

    log.append(sum_log)
    dx_log.append(sum_dx_log)

    W_log.append(W.copy())
    B_log.append(B.copy())
    
    W -= sum_dW * learning_rate
    B -= sum_db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,  line_error,
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
X = np.array([-1.5, -0.5])
y = np.array([0, 1])

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
        dZ = A - y[j]           #dL/dZ
        sum_dW += X[j] * dZ     #dL/dW
        sum_db += dZ            #dL/dB

    log.append(sum_log)
    dx_log.append(sum_dx_log)

    W_log.append(W.copy())
    B_log.append(B.copy())
    
    W -= sum_dW * learning_rate
    B -= sum_db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,  line_error,
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
X = np.array([0.4, 0.7])
y = np.array([0, 1])


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
        dZ = A - y[j]           #dL/dZ
        sum_dW += X[j] * dZ     #dL/dW
        sum_db += dZ            #dL/dB

    log.append(sum_log)
    dx_log.append(sum_dx_log)

    W_log.append(W.copy())
    B_log.append(B.copy())
    
    W -= sum_dW * learning_rate
    B -= sum_db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,  line_error,
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
X = np.array([-1, 1])
y = np.array([0.4, 0.6])

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
        dZ = A - y[j]           #dL/dZ
        sum_dW += X[j] * dZ     #dL/dW
        sum_db += dZ            #dL/dB

    log.append(sum_log)
    dx_log.append(sum_dx_log)

    W_log.append(W.copy())
    B_log.append(B.copy())
    
    W -= sum_dW * learning_rate
    B -= sum_db * learning_rate

    #Graphique Update
    update_graph(
        axes,
        line_log, line_dx,
        line_W, line_B,
        line_model, scatter_target,  line_error,
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