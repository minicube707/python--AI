
import  numpy as np


#Fonction
def sigmoïde(X):
    return 1 / (1 + np.exp(-X))

def relu(X):
    return np.where(X < 0, 0, X)

def ouput_shape(input_size, k_size, padding, stride):
    return np.int8((input_size - k_size + 2*padding)/stride +1)

def initialisation(i_size, k_size, padding, stride):
    o_size = ouput_shape(i_size.shape[0], k_size, padding, stride)

    K = np.random.randn(k_size, k_size)
    b = np.random.randn(o_size, o_size)

    return K, b

def model(X, K, b, f_activation):

    k_size = K.shape[0]
    b_size = b.shape[0]
    new_grid = np.zeros((b_size, b_size))
    
    k = K.flatten()
    for i in range(0, X.shape[0]-1):
        for j in range(0, X.shape[1]-1):
            x = X[i:i + k_size, j:j + k_size].flatten()
            new_grid[i, j] = np.correlate(k, x, mode="valid")[0]
            
    new_grid += b
    return f_activation(new_grid)

def log_loss(A, y):
    return  -1/y.size * np.sum( y*np.log(A) + (1-y)*np.log(1-A))

def gradients(X, y_pred, y_true):

    n = y_pred.size
    dY = 1/n - (1 - y_true)/(1 - y_pred) - (y_true)/(y_pred)
    dy_size =  dY.shape[0]

    dK = np.zeros((dy_size, dy_size))
    dy = dY.flatten()
    for i in range(0, X.shape[0]-1):
        for j in range(0, X.shape[1]-1):
            x = X[i:i + dy_size, j:j + dy_size].flatten()
            dK[i, j] = np.correlate(dy, x, mode="valid")[0]

    return (dK, dY)

def update(dY, db, K, b, learning_rate):
    K = K - learning_rate*dY
    b = b - learning_rate*db
    return (K, b)


X = np.array([[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]])

y = np.array([[0, 1], 
              [1, 1]])

#Initialisation
print("")
W, b = initialisation(X, 2, 0, 1)
print("Dimension de W", W.shape)
print("W =\n",W)
print("\nDimension de b ", b.shape)
print("b =\n",b)

#Model
print("")
A = model(X, W, b, sigmoïde)
print("Dimension de A ", A.shape)
print("A =\n",A)

#Log_Loss
print("")
l = log_loss(A, y)
print("loss =",l)

#Gradients
print("")
dW, db = gradients(X, A, y)
print("Dimension de dW ", dW.shape)
print("dW =\n",dW)
print("db =\n",db)

#Update
print("")
learning_rate = 0.01
W, b = update(dW, db, W, b, learning_rate)
print("Dimension de W ", dW.shape)
print("W =\n",W)
print("\nDimension de b ", b.shape)
print("b =\n",b)