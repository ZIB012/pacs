from .nn import NN
from .random_fnn import partioned_random_FNN
from .. import activations
from .. import initializers
from .. import regularizers
from ...backend import tf
import numpy as np

def psi_a(x):
    arr = np.ones((x.shape[0],1), dtype='float32')
    for i in range(x.shape[0]):
        if x[i] < -5/4 or x[i] > 5/4:
            arr[i] = 0
        elif x[i] >= 3/4 and x[i] <= 5/4:
            arr[i] = (1 - np.sin(2*np.pi*x[i]))/2
    return arr

def psi_b(x):
    arr = np.ones((x.shape[0],1), dtype='float32')
    for i in range(x.shape[0]):
        if x[i] < -5/4 or x[i] > 5/4:
            arr[i] = 0
        elif x[i] >= -5/4 and x[i] <= -3/4:
            arr[i] = (1 + np.sin(2*np.pi*x[i]))/2
        elif x[i] >= 3/4 and x[i] <= 5/4:
            arr[i] = (1 - np.sin(2*np.pi*x[i]))/2
    
    return arr

def psi_c(x):
    arr = np.ones((x.shape[0],1), dtype='float32')
    for i in range(x.shape[0]):
        if x[i] < -5/4 or x[i] > 5/4:
            arr[i] = 0
        elif x[i] >= -5/4 and x[i] <= -3/4:
            arr[i] = (1 + np.sin(2*np.pi*x[i]))/2
        
    return arr

def indicatrice_a(a,b):
    return lambda x: psi_a((2*x-b-a)/(b-a))

def indicatrice_b(a,b):
    return lambda x: psi_b((2*x-b-a)/(b-a))

def indicatrice_c(a,b):
    return lambda x: psi_c((2*x-b-a)/(b-a))

def partition_of_unity(npart, geom, data, layer_size, activation, initializer, Rm, b=0.0005):

    arr = np.linspace(geom.l, geom.r, npart + 1)
    nn_indicatrici = [indicatrice_b(arr[i], arr[i+1]) for i in range(1,npart-1)]
    nn_indicatrici.append(indicatrice_c(arr[-2], arr[-1]))
    nn_indicatrici.insert(0, indicatrice_a(arr[0], arr[1]))

    train_indicatrici = [nn_indicatrici[i](data.train_x) for i in range(npart)]
    test_indicatrici = [nn_indicatrici[i](data.test_x) for i in range(npart)]

    net = partioned_random_FNN(layer_size, activation, initializer, npart, nn_indicatrici, train_indicatrici, test_indicatrici, Rm=Rm, b=b)

    return net

