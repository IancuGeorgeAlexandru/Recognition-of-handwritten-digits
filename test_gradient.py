import numpy as np
import pandas as pd
import matplotlib.pylab as plt  

class ANN:
    def __init__(self, layers_size,input_layer_node):
        self.layers_size = layers_size
        #print("Layers size: ")
        #print(self.layers_size)
        self.parameters = {}

        # Numar total de straturi
        self.L = len(self.layers_size)
        #print("L: ")
        #print(self.L)

        # Exemple antrenament
        self.n = 0
        self.batch = 0

        # Urmarire de costuri
        self.costs = []
        self.layers_size.insert(0,input_layer_node)
        self.initialize_parameters()
 
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
 
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
 
    # Initializam parametrii aici pentru fiecare layer al reteleii in mod random dupa o distributie 
    # standard, media = 0, deviatia standard = 1
    def initialize_parameters(self):
        np.random.seed(1)
 
        for l in range(1, len(self.layers_size)):
            # Creez o matrice cu dimensiunile respective
            # Imapart la patratul dimensiunii stratului anterior pentru a normaliza 
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
 
    # Propagare inainte 
    def forward(self, X):
        store = {}
 
        A = X.T
        # Parcurgerea straturilor ascunse pana la penultimul 
        for l in range(self.L - 1):
            # Calculul valorilor ponderate
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            # Aplicam functia de activare
            A = self.sigmoid(Z)
            # Stocarea valorilor in dictionar
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z
 
        # Avem aici stratul de iesire unde folosim functia softmax
        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.softmax(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z
 
        return A, store
 
    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)
 
    # Propagare inapoi pentru a afla derivatele
    def backward(self, X, Y, store):
 
        derivatives = {}
 
        store["A0"] = X.T
 
        A = store["A" + str(self.L)]
        # Calculul dintre activarile obtinute si valorile etichetelor reale
        dZ = A - Y.T

        # Calculul derivatei in raport cu ponderile 
        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.batch

        # Calculul derivatei in raport cu bias-ul
        db = np.sum(dZ, axis=1, keepdims=True) / self.batch
        dAPrev = store["W" + str(self.L)].T.dot(dZ)
 
        # Se stocheaza derivatele
        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db
 
        # Se parcurg straturile de la ultimul pana la primul
        for l in range(self.L - 1, 0, -1):
            # Se calculeaza derivatele in raport cu stratul curent
            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            dW = 1. / self.batch * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.batch * np.sum(dZ, axis=1, keepdims=True)
            # Daca nu sunt la primul, se calculeaza in raport cu stratul anterior
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)
    
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
 
        return derivatives

    # Folosim aici bini-batch-uri pentru a le grupa pe rand
    # putem seta noi dimensiunea
    # este mai rapid asa si nici nu trebuie stocat tot setul de date
    def create_mini_batches(self,X, y, batch_size):
        
        mini_batches = []
        # Concatenare date
        data = np.hstack((X, y))
        np.random.shuffle(data)
        # Numarul de batch-uri
        n_minibatches = data.shape[0] // batch_size
        i = 0
     
        for i in range(n_minibatches + 1):
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_mini = mini_batch[:, :-10]
            Y_mini = mini_batch[:, -10:].reshape((-1, 10))
            mini_batches.append((X_mini, Y_mini))
        # Verificare sa vedem daca sunt date ramase, deci daca nu incap toate perfect
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-10]
            Y_mini = mini_batch[:, -10:].reshape((-1, 10))
            mini_batches.append((X_mini, Y_mini))
        return mini_batches
 
    # Antrenam modelul SGD
    def fit(self, X, Y, learning_rate=1, n_iterations=10,batch=32):
        np.random.seed(1)
        self.batch = batch

        for loop in range(n_iterations):
            mini_batches = self.create_mini_batches(X, Y, self.batch)

            # Pierdere si acuratete
            loss = 0
            acc = 0

            # Iterarea prin mini-batch-uri
            for mini_batch in mini_batches:

                # Separarea inapoi a datelor
                X_mini, y_mini = mini_batch

                # Propagarea inainte pentru a obtine activarile
                A, store = self.forward(X_mini)

                # Costul de pierdere prin entropie incrucisata
                loss += -1*np.mean(y_mini * np.log(A.T+ 1e-8))

                # Propagarea inapoi pentru a obtine derivatele
                derivatives = self.backward(X_mini, y_mini, store)
     
                # Mergem prin straturi pentru a face update
                for l in range(1, self.L + 1):
                    self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                        "dW" + str(l)]
                    self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                        "db" + str(l)]

                acc += self.predict(X_mini, y_mini)

            self.costs.append(loss)
            print("Epoch",loop+1,"\steps ",len(mini_batches),"Train loss: ", "{:.4f}".format(loss/len(mini_batches)),
                                                "Train acc:", "{:.4f}".format(acc/len(mini_batches)))
                    
                
                    
    # Salvam ponderile local pentru a nu rula mereu antrenarea
    def save_weights(self):
        np.save("weights.npy",ann.parameters ,allow_pickle=True)
    
    # Incarcam ponderile 
    def load_weights(self,dir):
        weights=np.load(dir,allow_pickle=True)
        for i in ann.parameters.keys():
            ann.parameters[str(i)] = weights.item().get(str(i))
            
        print('Weight loaded')
        
    # Prezicem aici o anumita poza, data ca index
    def predict(self, X, Y):
        A, cache = self.forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy
 
    # Afisam evolutia costului
    def plot_cost(self):
        #plt.figure()
        #plt.plot(np.arange(len(self.costs)), self.costs)
        #plt.xlabel("epochs")
        #plt.ylabel("Loss")
        #plt.grid()
        #plt.show()
        return self.costs

 
# Transformarea din label de o dimensiunea in vector de 10 cu 1 pe pozitia label-ului initial
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    #one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Ajutatoare pentru incararea de date
def load_mnist():
    from keras.datasets import mnist
    (X, y), (X_test, y_test) = mnist.load_data()
    #Normalizare date
    X = X / 255.
    
    import numpy as np
    X =  np.reshape(X, (60000,-1))
    print(X.shape)
    y = one_hot(y)
    print(y.shape)
    
    return  X[1000:] , X[:1000] , y[1000:] , y[:1000] #split data into train test

# Rulare direct aici
if __name__ == '__main__':
    train_x , test_x , train_y , test_y = load_mnist()
    
    layers_dims = [10, 10]
    
    ann = ANN(layers_dims,train_x.shape[1])
    ann.fit(train_x, train_y, learning_rate=.1, n_iterations=100,batch=64)

    print("\nTrain Accuracy:", "{:.4f}".format(ann.predict(train_x, train_y)))
    print("Test Accuracy:", ann.predict(test_x, test_y))