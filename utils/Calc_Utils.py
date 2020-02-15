import numpy as np
from theano.tensor import shared_randomstreams
import theano.tensor as T
import theano

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def vectorized_result(j):
    print(j)
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def dropout_layer(layer, p_dropout):

    # create a seeded random stream for binomials
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999)) # seed = 0 with highest integer = 999999
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape) # matrix of either 1 or 0
    return layer*T.cast(mask, theano.config.floatX)  # randomly kill a lot of neurons, setting their affect to 0;

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return a-y


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


