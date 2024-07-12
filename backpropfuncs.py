import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk

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
    #print(f"Layer index = {layer_index}")
    transpose_weights = np.transpose(network.weights[layer_index+1])
    #print(f"Transpose weights shape: {np.shape(transpose_weights)}\n"
          #f"next layer ({layer_index+1} error shape: {np.shape(network.errors[layer_index+1])}")
    error = np.matmul(transpose_weights, network.errors[layer_index+1])
    #print(f"Layer {layer_index} shape: {np.shape(network.layers[layer_index])}")
    relu_grad = relu_prime(network.layers[layer_index])
    error = np.multiply(error, relu_grad)
    return error


def find_weight_grad(network, layer_index):
    # weights are organized as a j x k matrix
    # k = size(layer-1), j = size(layer)
    # matmul(layer-1, error(layer) would give a kxj matrix
    activations = network.layers[layer_index-1]
    error = network.errors[layer_index]
    gradient = np.outer(error, activations)
    return gradient


def read_image(image_number):
    with open("C:/Users/Timmy/Downloads/MNIST_ORG/train-images.idx3-ubyte", mode = "r+b") as file:
        file.seek(16+image_number*784)
        pixels = np.fromfile(file, dtype=np.uint8, count = 784) / 255
        return pixels


def read_label(label_number):
    correct_output_array = np.zeros(10)
    with open("C:/Users/Timmy/Downloads/MNIST_ORG/train-labels.idx1-ubyte", mode = "r+b") as file:
        file.seek(8+label_number)
        value = int.from_bytes(file.read(1), byteorder='big')
        correct_output_array[value] = 1
        return correct_output_array

def load_all_images():
    with open("C:/Users/Timmy/Downloads/MNIST_ORG/train-images.idx3-ubyte", mode = "r+b") as file:
        file.seek(16)
        image_array = np.fromfile(file, dtype = np.uint8, count = 60000*784) / 255
        return image_array.reshape(60000, 784)

def load_all_test_images():
    with open("C:/Users/Timmy/Downloads/MNIST_ORG/t10k-images.idx3-ubyte", mode = "r+b") as file:
        file.seek(16)
        image_array = np.fromfile(file, dtype = np.uint8, count = 10000*784) / 255
        return image_array.reshape(10000, 784)

def load_all_test_labels(one_hot_or_nums = "one_hot"):
    with open("C:/Users/Timmy/Downloads/MNIST_ORG/t10k-labels.idx1-ubyte", mode = "r+b") as file:
        file.seek(8)
        label_array = np.fromfile(file, dtype=np.uint8, count = 10000)
        if(one_hot_or_nums == "nums"):
            return label_array
        one_hot_labels = np.zeros((10000, 10))
        one_hot_labels[np.arange(10000), label_array] = 1
        return one_hot_labels


def load_all_labels(one_hot_or_nums = "one_hot"):
    with open("C:/Users/Timmy/Downloads/MNIST_ORG/train-labels.idx1-ubyte", mode = "r+b") as file:
        file.seek(8)
        label_array = np.fromfile(file, dtype=np.uint8, count = 60000)
        if(one_hot_or_nums == "nums"):
            return label_array
        one_hot_labels = np.zeros((60000, 10))
        one_hot_labels[np.arange(60000), label_array] = 1
        return one_hot_labels


def is_correct(expected, output):
    should_be_highest = output[expected]
    for value in output:
        if value > should_be_highest:
            return False
    return True

def to_file(values, path):
    file = open(path, mode = 'w')
    file.write(values)




