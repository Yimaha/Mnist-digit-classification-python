
import numpy as np
import random
from utils import Calc_Utils as Calc, Read_Util as Read


class Network:
    def __init__(self, size):    # size is a 1D array
        self.num_layer = len(size)
        self.size = size  # shape
        self.weight = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]  # weight
        self.bias = [np.random.randn(y, 1) for y in size[1:]]  # bias

    def feed_forward(self, a):
        for weight, bias in zip(self.weight, self.bias):
            a = Calc.sigmoid(np.dot(weight, a) + bias)
        return a

    def stochastic_gradient_descent(self, training_set, epochs, mini_batch_size, eta, test_data=None):
        n = len(training_set)
        for j in range(epochs):
            print("start training in epochs {0}".format(j))
            random.shuffle(training_set)
            mini_batches = [training_set[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("training for this round is finished")
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batches, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        # for mini_batch in mini_batches:
        for input, output in mini_batches:
            delta_nabla_w, delta_nabla_b = self.backprop(input, output)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weight = [w - (eta/len(mini_batches))*nw for w, nw in zip(self.weight, nabla_w)]
        self.bias = [b - (eta/len(mini_batches))*nb for b, nb in zip(self.bias, nabla_b)]

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_result)

    def backprop(self, mini_batch_input, mini_batch_output):
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        activation = mini_batch_input   # current activision
        activations = [mini_batch_input]    # array to store all activisions
        zs = []     # stored all output (basically after sigmoid)
        for w, b in zip(self.weight, self.bias):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Calc.sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], mini_batch_output) * Calc.sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta
        for L in range(2, self.num_layer):
            z = zs[-L]
            sp = Calc.sigmoid_prime(z)
            delta = np.dot(self.weight[-L+1].transpose(), delta) * sp
            nabla_b[-L] = delta
            nabla_w[-L] = np.dot(delta, activations[-L-1].transpose())
        return nabla_w, nabla_b

    def cost_derivative(self, output, expected_output):
        return output - expected_output



# play ground

network = Network([784, 100, 30, 10])
training, validate, test = Read.load_data()
network.stochastic_gradient_descent(list(training), 100, 10, 5.0, list(test))
