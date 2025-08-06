
import numpy as np

"""
============================
==========Fonction==========
============================
"""
"""
sigmoïde:
=========DESCRIPTION=========
Apply the sigmoide function at the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def sigmoide(X):
    return 1/(1 + np.exp(-X))


"""
relu:
=========DESCRIPTION=========
Apply the relu function at the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def relu(X):
    return np.where(X < 0, 0, X)


"""
max_pooling:
=========DESCRIPTION=========
Return the max of each row of the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def max_pooling(X):
    a = np.int8(np.sqrt(X.shape[1]))
    return np.max(X, axis=2).reshape((X.shape[0], a, a))


"""
softmax:
=========DESCRIPTION=========
Apply the softmax function at the activation function

=========INPUT=========
numpy.array     X :     the activation matrice

=========OUTPUT=========
numpy.array     x :     array containe the next activation
"""
def softmax(X):
    res = np.array([])
    for i in range(X.shape[0]):
        res = np.append(res, np.exp(X[i,:]) / np.sum(np.exp(X[i,:])))
         
    return res.reshape((X.shape))


"""
correlate
=========DESCRIPTION=========
Perform a correlation between two arrays (activation and kernel).

=========INPUT=========
A (np.ndarray): Activation matrix (shape: [in_channels, ...])
K (np.ndarray): Kernel matrix (shape: [out_channels, kernel_size])
b (np.ndarray): Bias vector (shape: [out_channels])
x_size (int): Size of the spatial dimension of the activation

=========OUTPUT=========
Z_concat (np.ndarray): Next activation array (shape: [out_channels, x_size, x_size])
"""
def correlate(A, K, b, x_size):

    # Liste pour stocker chaque couche transformée
    L_A, NB_Dot_Product, K_Size =  A.shape
    NB_K, L_K, K_Size, one = K.shape

    Z = np.zeros((NB_K, NB_Dot_Product, one))

    #For each kernel
    for i in range(NB_K):
        
        #For each activation
        for j in range(L_A):
            
            Z[i] = A[j].dot(K[i, j])
            
    Z += b    
    Z = Z.reshape((NB_K, x_size, x_size))
    Z = np.clip(Z, -88, 88)

    return Z


"""
convolution:
=========DESCRIPTION=========
Do the full convolution of two arrays

=========INPUT=========
numpy.array     dZ :            the derivated of the previous activation (what should be the activation)
numpy.array     K :             the kernel matrice
int             k_size_sqrt :   the size in row of the kernel

=========OUTPUT=========
numpy.array    next_dZ :       Array containe the derivated for the next layer
"""
def convolution(dZ, K, k_size_sqrt):
     
    #new_dz is intput with a pas to do the the cross product with all value
    new_dZ = np.pad(dZ, pad_width=((0, 0), (k_size_sqrt - 1, k_size_sqrt - 1), (k_size_sqrt - 1, k_size_sqrt - 1)), mode='constant', constant_values=0)

    #next_dz is the output
    next_dZ = np.zeros((K.shape[1], dZ.shape[1]+k_size_sqrt-1, dZ.shape[2]+k_size_sqrt-1))
    
    #For each kernel
    for a in range(K.shape[0]):
        
        #Select the correct layer from the DZ & kernel
        dZ_layer = new_dZ[a]
        tensor_K = K[a]
        
        #Copy and concat the DZ to match the size of the kernel
        dZ_layer = np.repeat(dZ_layer[np.newaxis, :, :], repeats=tensor_K.shape[0], axis=0)

        #Do the convolution
        #FOR EACH LAYER
        for b in range(next_dZ.shape[0]):

            #FOR SELCTION COLOMN
            for c in range(next_dZ.shape[1]):

                #FOR SELCTION ROW
                for d in range(next_dZ.shape[2]): 

                    #DO THE CONVOLUTION
                    next_dZ[b, c, d] += np.dot(dZ_layer[b, c:c + k_size_sqrt, d:d + k_size_sqrt].flatten(), tensor_K[b][::-1].flatten())

    return next_dZ