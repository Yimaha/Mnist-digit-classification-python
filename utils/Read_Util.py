import numpy as np
import pickle
import gzip
import os

def get_project_root():
    """Returns project root folder."""
    return os.path.dirname(os.path.abspath(__file__)).split('utils')[0]

def read_raw():
    buffer = gzip.open(get_project_root() + 'data/mnist.pkl.gz', 'rb')
    training_set, validation_data, test_data = pickle.load(buffer, encoding="latin1")
    buffer.close()
    return training_set, validation_data, test_data


def vectorized_result(input):
    e = np.zeros((10, 1))
    e[input] = 1.0
    return e


def processed_data(set):
    input = [np.reshape(x, (784, 1)) for x in set[0]]
    result = [vectorized_result(y) for y in set[1]]
    return zip(input, result)


# actual read function, others are helper
def load_data():
    training, validate, test = read_raw()
    return processed_data(training), processed_data(validate), processed_data(test)
