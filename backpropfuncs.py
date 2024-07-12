import numpy as np


def softmax(x):
    e_x = np.exp(x - max(x))
    return e_x / e_x.sum(axis=0)


def relu_prime(x):
    return np.where(x > 0, 1, 0)


def output_layer_error(output_layer, label):
    # cross entropy loss, softmax activation for last layer
    # label is a one-hot vector with shape(output_layer)
    error = np.subtract(output_layer, label)
    return error


def error_from_next_layer(network, layer_index):
    transpose_weights = np.transpose(network.weights[layer_index])
    error = np.matmul(transpose_weights, network.errors[layer_index + 1])
    relu_grad = relu_prime(network.layers[layer_index])
    error = np.multiply(error, relu_grad)
    return error


def find_weight_grad(network, layer_index):
    # weights are organized as a j x k matrix
    # k = size(layer-1), j = size(layer)
    # matmul(layer-1, error(layer) would give a kxj matrix
    columns = []
    for activation in network.layers[layer_index-1]:
        columns.append(activation * network.errors[layer_index])
    gradient = np.hstack(columns)
    return gradient

