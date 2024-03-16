import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from test_gradient import *


from keras.datasets import mnist

# Luam setul de date
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Ii dam dimensiunea acceptata de clasa ANN
X_train = X_train.reshape(60000, 784) 
X_test = X_test.reshape(10000, 784) 

y_train = one_hot(y_train)
y_test = one_hot(y_test)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# Il facem cu virgula pentru a putea normaliza
X_train = X_train.astype('float32')   
X_test = X_test.astype('float32')

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

# Normalizam informatia
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# Aici reducem informatia pentru a vedea diferentele
from sklearn.decomposition import PCA
pca = PCA(n_components=.85)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Observam noua informatie
print(f'train_img shape : {X_train_pca.shape}')
print(f'test_img shape : {X_test_pca.shape}')

# Am folosit alt clasificator
# merge mai incet dar are aceeasi acuratete
# merge mult mai bine pentru PCA

"""
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver = 'lbfgs')

start_time = time.time()
clf = MLPClassifier(solver = 'lbfgs', max_iter=100)
clf.fit(X_train, y_train)
"""

# Aici comparam inaginile, inainte de PCA si dupa PCA
"""
index = 5000

current_image = X_test[None,index,:]
current_image = current_image.reshape((28, 28))
plt.gray()
plt.imshow(current_image, cmap = 'binary', interpolation='nearest')
plt.show()
current_image_pca = X_test_pca[None,index,:]
current_image_pca = pca.inverse_transform(current_image_pca)
current_image_pca = current_image_pca.reshape((28, 28))
plt.gray()
plt.imshow(current_image_pca, cmap = 'binary', interpolation='nearest')
plt.show()
"""

# Aici declaram straturile ascunse de dimensiune 10
layers_dims = [10,10]


# Aici facem cele doua teste si le comparam
print('='*60)
print(f'Control result with original dataset (100% variance)')
start_time = time.time()

ann = ANN(layers_dims,X_train.shape[1])
ann.fit(X_train, y_train, learning_rate=.1, n_iterations=100,batch=64)

print("\nTrain Accuracy:", "{:.4f}".format(ann.predict(X_train,y_train)))
print("Test Accuracy:", ann.predict(X_test, y_test))
costs1 = ann.plot_cost()
end_time = time.time()

print(f'Time (seconds) : {end_time - start_time} \n')

print('='*60)
print(f'Control result with PCA dataset (90% variance)')
start_time = time.time()

layers_dims = [10,10]
ann = ANN(layers_dims,X_train_pca.shape[1])
ann.fit(X_train_pca, y_train, learning_rate=.05, n_iterations=100,batch=64)

print("\nTrain Accuracy:", "{:.4f}".format(ann.predict(X_train_pca, y_train)))
print("Test Accuracy:", ann.predict(X_test_pca, y_test))
#ann.plot_cost()
costs2 = ann.plot_cost()
end_time = time.time()

print(f'Time (seconds) : {end_time - start_time} \n')

plt.figure()
plt.plot(np.arange(len(costs1)), costs1)
plt.plot(np.arange(len(costs2)), costs2)
plt.legend(["Fara PCA", "Dupa PCA"])
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()