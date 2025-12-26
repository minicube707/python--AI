import pickle
import numpy as np
import os

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

def load_batch(path):
    print("SEARCH ", path)
    with open(path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    X = batch[b'data']
    y = batch[b'labels']
    return X, y

data_dir = r"C:\Users\User\Downloads\cifar-10-python\cifar-10-batches-py"  # adapte si besoin

X_list = []
y_list = []

for i in range(1, 6):
    X, y = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
    X_list.append(X)
    y_list.append(y)

X, y = load_batch(os.path.join(data_dir, "test_batch"))
X_list.append(X)
y_list.append(y)

X = np.concatenate(X_list, axis=0)  # (60000, 3072)
y = np.concatenate(y_list, axis=0)  # (60000,)

# (N, 3072) → (N, 3, 32, 32)
X = X.reshape(-1, 3, 32, 32)

# (N, 3, 32, 32) → (N, 32, 32, 3)
X = X.transpose(0, 2, 3, 1)


X = X.astype("float32") / 255.0
y = np.array(y, dtype=np.int64)

np.savez("cifar10_train.npz", data=X, target=y)

d = np.load("cifar10_train.npz")
print(d["data"].shape)    # (60000, 32, 32, 3)
print(d["target"].shape)  # (60000,)
print(d["data"].dtype)    # float32