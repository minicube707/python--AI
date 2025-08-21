
import  numpy as np
import  matplotlib.pyplot as plt

#Fonction
def sigmoïde(X):
    return 1/(1 + np.exp(-X))

def relu(X):
    return np.where(X < 0, 0, X)


def ouput_shape(input_size, k_size, padding, stride):
    return np.int8((input_size - k_size + 2*padding)/stride +1)

def initialisation(i_size, k_size, padding, stride):
    o_size = ouput_shape(i_size.shape[0], k_size, padding, stride)

    K = np.random.randn(k_size**2, 1)
    b = np.random.randn(o_size**2, 1)

    return K, b

def model(X, K, b, f_activation):

    b_size = b.shape[0]
    new_grid = np.zeros((b_size, 1))
    new_grid = np.dot(X.T, K)
    new_grid += b

    return f_activation(new_grid).reshape((2,2))

def log_loss(A, y):
     return  -1/y.size * np.sum( y*np.log(A) + (1-y)*np.log(1-A))

def gradients(X, y_pred, y_true):

    dY = y_pred - y_true
    dy_size =  dY.shape[0]

    dK = np.zeros((dy_size**2, 1))
    dy = dY.reshape((1, dY.size))
    for i in range(0, 2): #X.shape[0]-1
            dK[i] = np.dot(dy, X[i])

    return (dK, dy.T)

def update(dY, db, K, b, learning_rate):
    K = K - learning_rate*dY
    b = b - learning_rate*db
    return (K, b)

def reshape(X, K):

    k_size = K.size
    k_size_sqrt = np.int8(np.sqrt(k_size))
    new_X = np.array([])
    
    for i in range(0, X.shape[0]-1):
        for j in range(0, X.shape[1]-1):
            new_X = np.append(new_X, X[i:i + k_size_sqrt, j:j + k_size_sqrt])

    
    return new_X.reshape(((X.shape[0]-1)*(X.shape[1]-1), k_size))

X = np.array([[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]])

y = np.array([[0, 1], 
              [1, 0]])

#Initialisation
learning_rate = 0.01
nb_iteration = 10000


K, b = initialisation(X, 2, 0, 1)

X = reshape(X, K)
l_array = np.array([])

for _ in range(nb_iteration):

    A = model(X, K, b, sigmoïde)
    l_array = np.append(l_array, log_loss(A, y))
    dK, db = gradients(X, A, y)
    K, b = update(dK, db, K, b, learning_rate)


print("\nFinal activation\n",A)
print("\nKernel\n",K)
print("\nBias\n",b)

plt.figure()
plt.title("Evolution du cout en fonction des itérations")
plt.plot(l_array)
plt.show()

plt.figure()
plt.title("Kernel")
plt.imshow(K.reshape((2,2)), cmap="gray")
plt.colorbar()
plt.axis("off")
plt.show()

plt.figure()
plt.title("Bias")
plt.imshow(b.reshape((2,2)), cmap="gray")
plt.colorbar()
plt.axis("off")
plt.show()

plt.figure()
plt.title("Y prediction")
plt.imshow(A.reshape((2,2)), cmap="gray")
plt.colorbar()
plt.axis("off")
plt.show()
