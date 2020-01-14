import numpy as np
import random
import sys
import json
from utils import Calc_Utils as Calc,  Read_Util as Read
from datetime import datetime

class Network2(object):
    def __init__(self, sizes, cost=Calc.CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # old declaration method for weight that tends to produce large weights.
    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feed_forward(self, a):
        for weight, bias in zip(self.weights, self.biases):
            a = Calc.sigmoid(np.dot(weight, a) + bias)
        return a

    def stochastic_gradient_descent(self,
                                    training_set,
                                    kill_limit,
                                    mini_batch_size,
                                    eta,
                                    lmbda=0.0,
                                    test_data=None,
                                    monitor_evaluation_cost=False,
                                    monitor_evaluation_accuracy=False,
                                    monitor_training_cost=False,
                                    ):

        if(test_data): n_test = len(test_data)
        n = len(training_set)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        j = 0
        maximum_accuracy = 0
        accuracy_no_increase_counter = 0
        while True:
            print("start training in epochs {0}".format(j))
            random.shuffle(training_set)
            mini_batches = [training_set[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            print("training for this round is finished")
            if monitor_training_cost:
                cost = self.total_cost(training_set, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_evaluation_cost:
                cost = self.total_cost(test_data, lmbda)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(test_data, convert=True)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_test));

            print("Epoch {0} complete".format(j))
            j = j + 1
            accuracy_no_increase_counter = accuracy_no_increase_counter + 1
            accuracy = self.accuracy(training_set, convert=True)
            training_accuracy.append(accuracy)
            print("Accuracy on training data: {} / {}".format(accuracy, n))
            if accuracy > maximum_accuracy:
                maximum_accuracy = accuracy
                accuracy_no_increase_counter = 0
            elif accuracy_no_increase_counter == kill_limit:
                break

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [(1 - eta * lmbda / n) * w - (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / len(mini_batch)) * nb
                           for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = Calc.sigmoid(z)
            activations.append(activation)
        # backward pass
        # zs[-1] = without sigmoid
        # activation[-1] = with sigmoid
        # y = expected output
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = Calc.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feed_forward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feed_forward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        # interesting enough, the data set I am working with seems to have no requirement on convert.
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        # normal regularization term from weights
        for x, y in data:
            a = self.feed_forward(x)
            if convert: y = Calc.vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        # the L2 regularization term
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(Calc, data["cost"])
    net = Network2(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


now = datetime.now()

network = Network2([784, 100, 30, 10])
training, validate, test = Read.load_data()
network.stochastic_gradient_descent(list(training),
                                    10,
                                    10,
                                    0.5,
                                    lmbda=5.0,
                                    test_data=list(test))
network.save("trained_" + now.strftime("%Y-%m-%d-%H-%M-%S"))
print("finished")