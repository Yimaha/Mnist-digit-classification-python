#### Libraries
# Standard library
from utils import Calc_Utils as Calc, Read_Util as Read

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample


# Activation functions for neurons.
def linear(z): return z


def ReLU(z): return T.maximum(0.0, z)


from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

# Constants
GPU = True
if GPU:
    try:
        theano.config.device = 'gpu'
    except:
        pass  # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU")


# Load the MNIST data
def load_data_shared():
    training_data, validation_data, test_data = Read.load_data()

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return [shared(training_data), shared(validation_data), shared(test_data)]


# start with implementing different types of layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_function=sigmoid, p_dropout=0.0):
        self.n_in = n_in  # amount of input neuron
        self.n_out = n_out  # amount of output neuron
        self.activation_function = activation_function  # the neuron activation function, can be any or the default which is sigmoid
        self.p_dropout = p_dropout  # probability of a neuron to be added to the dropout network, default to be 0 so all neuron are included

        '''
        NOTE: Why is standard deviation sqrt of a/n_out?
        
        If you are not using sigmoid neurons, it makes no different,
        if you are using sigmoid neurons, it makes the sigmoid(z) of neuron less close to either 0 / 1, so the 
        training speed is much faster. (when the sigmoid neuron is saturated, usually it has a small gradient
        and slows down the training substantially)
        
        HOWEVER, given enough epochs, the result would vary only slightly since a small gradient is still a gradient. 
        '''

        self.weight = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX,
            ),
            name='weight',
            borrow=True
        )

        '''
        NOTE: Unlike weight, bias don't need sigmoid special case because it constantly rely on 1 parameter
        '''

        self.bias = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=1.0, size=(n_out, 1)),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.params = [self.weight, self.bias]

    def set_input(self, input_array, input_dropout, mini_batch_size):
        # the non-dropout result, used for validation / testing
        # compute all input of every mini_batch at the same time though matrix.
        self.input_matrix = input_array.reshape((mini_batch_size, self.n_in))
        ''' 
        NOTE: (1 - self.p_dropout) since when validating, we have to assume that we only take a 
        fraction of each network's thinking into one. 
        
        For example, if input_dropout = 0.9, we take only 
        0.1 of each neuron's output because they were trained to be the only neuron exists in the network out of 10
        other neurons. Now imagine if you are validating, which means you are listening to all 10 neuron whe were trained
        to be the only neuron, you need to take only 1/10 of each output so it add up to a full output instead of 
        10x the original output., 
        '''
        self.output = self.activation_function(
            (1 - self.p_dropout) * T.dot(self.input_matrix, self.weight) + self.bias
        )
        self.y_out = T.argmax(self.output, axis=1) # max of each row?

        # the dropout result, which is used for computation(learning)
        self.input_dropout = Calc.dropout_layer(
            input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout
        )
        self.output_dropout = self.activation_function(
            T.dot(self.input_dropout, self.weight) + self.bias
        )

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        '''
        NOTE: for non-sigmoid neuron initialization makes very little difference
        NOTE: softmax is best used for output
        '''
        self.weight = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.bias = theano.shared(
            np.zeros((n_out, 1), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.weight, self.bias]

    def set_input(self, input_array, input_dropout, mini_batch_size):
        # used softmax instead of the customized activation function.
        self.input_matrix = input_array.reshape((mini_batch_size, self.n_in))

        self.output = softmax(
            (1 - self.p_dropout) * T.dot(self.input_matrix, self.weight) + self.bias
        )
        self.y_out = T.argmax(self.output, axis=1)  # max of each row?

        # the dropout result, which is used for computation(learning)
        self.input_dropout = Calc.dropout_layer(
            input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout
        )
        self.output_dropout = softmax(
            T.dot(self.input_dropout, self.weight) + self.bias
        )

    def cost(self, net):
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y]) 

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))