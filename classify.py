import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
tf.compat.v1.disable_v2_behavior()
tf.TF_ENABLE_ONEDNN_OPTS=0
# Avem nevoie de metoda placeholder care in v2 nu mai exista
import input_data
# Clasa pentru citirea pozelor de pe site si dezarhivarea lor
import cv2
import numpy as np
import math
from scipy import ndimage

# Centreaza cum trebuie mutata imaginea dupa centrul de masa
def getBestShift(img):
    cy,cx = ndimage.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx,shifty

# Muta numarul pentru a-l pozitiona in centrul de masa
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def classifyfcn(filename):
    # creez aici un folder cu informatiile, doar daca este nevoie
    # one_hot = true inseamna ca fiecare imagine nu are un singur label cu clasa din care face parte
    # , ci are un vector de dimensiune 10 si exista o singura valoare 1, restul 0, in pozitia
    # egala cu clasa respectiva din care face parte
    # Exemplu pentru o poza care reprezinta 9:
    # [0,0,0,0,0,0,0,0,0,1]
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Reteaua noastra neuronala o sa aiba un graf de calcul, iar 
    # placeholder-ul permite introducerea datelor de tip imagine
    # avem date de tip float, dar avem un numar variabil de imagini, din cauza
    # lui 'none', deci batch size-ul poate varia, dar fiecare imagine are cate 
    # 28*28 = 784 pixeli 
    x = tf.placeholder("float", [None, 784])

    # Avem nevoie de o variabila in graficul de calcul, mai exact, ponderile (weights)
    # asociate conexiunilor dintre straturile retelei
    # Initializam variabila cu 0, care o sa fie optimizata pentru imbunatatirea performantelor
    # Avem 784 de neuroni in stratul de intrare si 10 in stratul de iesire, 
    # deoarece avem 10 clase
    W = tf.Variable(tf.zeros([784,10]))

    # Foarte similar, doar ca pentru bias, pentru un strat de 10 neuroni
    # acest termen se adauga la final
    b = tf.Variable(tf.zeros([10]))

    # Facem aici produsul scalar dintre vectorul de intrare x si 
    # matricea ponderilor W, declarata mai sus, la care se adauga bias-ul b
    # Aceasta operatie reprezinta o propagare inainte
    # Se aplica apoi softmax peste rezultat, care transforma rezultatele
    # intr-o distributie de probabilitate peste toate cele 10 clase,
    # urmand, apoi, sa alegem clasa cu probabilitatea cea mai mare
    y = tf.nn.softmax(tf.matmul(x,W) + b)


    # Aici regasim valorile reale pe care vrem sa le antrenam (numerele de la 0-9)
    # pentru un numar variabil de imagini
    y_ = tf.placeholder("float", [None,10])

    # Folosim functia entropiei incrucisate pe care vrem sa o nimimizam pentru a ne 
    # face modelul mai performant
    # Facem, teoretic, diferenta dintre distributia reala y_ si distributia prezisa de
    # model, pe care, evident vrem sa o minimizam pentru a ne apropia cat mai mult de 
    # valorile reale
    # Se foloseste functia logaritmica pentru o penalizare mai mare a prezicerilor eronate
    # Aducerea la suma si negativizare asigura ca costul este pozitiv si ca creste pe masura ce 
    # distributiile devin mai diferite
    # Reducem in acelasi timp dupa toate dimensiunile, deci obtinem un scalar
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    # Folosim optimizatorul GradientDescent pentru a minimiza entropia, care este 
    # functia de cost
    # Dupa gradientul descendent optimizam ponderile si bias-urile, cu o rata 
    # de invatare de 0.01
    # Ne ducem in directia in care se minimizeaza functia respectiva
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Initializam variabilele globale din retea
    init = tf.global_variables_initializer()

    # Incepem sesiunea
    sess = tf.Session()
    sess.run(init)

    # Folosim 5000 batch-uri a cate 50 de poze fiecare
    for i in range(5000):
        # Se obtin batch-uri mici de cate 50 de imagini din dataset
        # Cele cu x sunt imaginile efective
        # Ceele cu y sunt etichetele cu clasa din care face parte fiecare imagine
        batch_xs, batch_ys = mnist.train.next_batch(50)
        # Se ruleaza operatia de antrenament 'train_step' deci gradientul descendent si se trimit
        # valori pentru x si pentru y_, care sunt etichetele reale
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    """
        Trecem acum la partea de testare, unde ne trebuie datele de test din setul
        de date
    """

    # argmax returneaza indexul elementului dintr-un tensor care are cea mai
    # mare valoare pe o anumita axa
    # Daca cele doua valori sunt egale, element cu element din tensor,
    # deci valoarea prezisa de model si cea reala,
    # atunci avem o valoarea corect
    # Rezultatul este un boolean care spune daca rezultatul este acelasi sau nu
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    # Facem un cast aici de la true la 1.0 si de la false la 0.0 pentru a putea face media 
    # pe tot tensorul respectiv pentru a obtine acuratetea
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Printam aici acuratetea modelului pe setul de date dat si pe modelul pe care 
    # l-am invatat mai devreme
    print("Acuratetea modelului pe dataset-ul mare:")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    """
        Incepem partea de testare pe o anumita imagine data 
        de utilizator
    """

    # Ne definim aici imaginea. O singura imagine cu dimensiunea specificata
    images = np.zeros((1,784))
    # Avem aici si labelul de 10 elemente cu valorile corecte, de care nu avem noi nevoie 
    # aplicatia noastra, dar avem nevoie de el pentru input la functia modelului
    correct_vals = np.zeros((1,10))

    # Se poate modifica aici pentru a testa mai multe imagini si pentru a face acuratete 
    # separat pe ele
    i=0
    for no in [0]:
        # Citim imaginea data ca parametru aici sau imaginile
        #gray = cv2.imread("test_images/own_"+str(no)+".png", 0)
        gray = cv2.imread("custom_test_images/" + filename, 0)

        # Ii reducem dimensiunea la 28x28 de pixeli si o facem pe tonuri de gri
        gray = cv2.resize(255-gray, (28, 28))

        # Facem aici o biniarizare a imaginii in functie de o valoare de prag
        # Se foloseste metoda Otsu pentru a selecta pragul
        # 128 este pragul initial, inainte de metoda de mai sus
        # 255 reprezinta la cat se modifica valoarea pixelilor daca
        # valoarea lor anterioara depaseste pragul
        # Rezultatul este asadar, o imagine doar cu doua culori, alb si negru
        (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Pozele din dataset pot incapea chiar pe o imagine de 20 pe 20 de biti si sunt centrate
        # Trebuie sa facem acelasi lucru si cu pozele noastre

        # Eliminam aici toate liniile si coloanele care nu au nimic in ele
        # pentru a obtine doar numarul si fara altceva pe langa el, deci nici
        # chenar alb

        # Elimina linii negre de sus in jos cat timp nu e nimic pe ele
        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        # Elimina coloane negre de la stanga la dreapta cat timp nu e nimic pe ele
        while np.sum(gray[:,0]) == 0:
            gray = np.delete(gray,0,1)

        # Elimina linii negre de jos in sus cat timp nu e nimic pe ele
        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        # Elimina coloane negre de la dreapta la stanga cat timp nu e nimic pe ele
        while np.sum(gray[:,-1]) == 0:
            gray = np.delete(gray,-1,1)

        rows,cols = gray.shape

        # Dimensionam aici imaginea la 20x20 pixeli, dar respectan raportul initial
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            # 
            gray = cv2.resize(gray, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            # 
            gray = cv2.resize(gray, (cols, rows))

        # Redimensionam aici la 28x28 pixeli pentru a pastra simetria
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

        # Shift pentru a ajunge in centrul de masa
        shiftx,shifty = getBestShift(gray)
        shifted = shift(gray,shiftx,shifty)
        gray = shifted

        # Salvam imaginea
        cv2.imwrite("custom_processed_images/" + filename, gray)
        flatten = gray.flatten() / 255.0
        images[i] = flatten
        correct_val = np.zeros((10))
        correct_val[no] = 1
        correct_vals[i] = correct_val
        i += 1

    """
        Rezultatul este un vector cu numerele prezise
    """
    prediction = tf.argmax(y,1)
    """
        Rulam functia pentru predictie pentru a returna numarul prezis
    """
    return sess.run(prediction, feed_dict={x: images, y_: correct_vals})