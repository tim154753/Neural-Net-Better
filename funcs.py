import numpy as np
import backpropfuncs as bp

#layers size cannot change after network initialization


class Network:
    def __init__(self, *args, learning_rate = 0.1):
        self.layers = [np.zeros(arg) for arg in args]
        self.weights = [np.zeros((args[i+1], args[i])) for i in range(len(args)-1)]
        self.biases = [np.zeros(arg) for arg in args[1:]]
        self.weighted_inputs = [np.zeros(arg) for arg in args]
        self.errors = [np.zeros(arg) for arg in args[1:]]
        self.weight_gradients = [np.zeros((args[i+1], args[i])) for i in range(len(args)-1)]
        self.bias_gradients = [np.zeros(arg) for arg in args[1:]]
        self.num_layers = len(args)
        self.learning_rate = learning_rate

    def __str__(self):
        result = "This is the network"
        for i, layer in enumerate(self.layers):
            result += f"\n Layer {i} values: {layer}\n"
        for weights, biases in zip(self.weights, self.biases):
            result += f"\nWeights:\n--{weights}\nBiases:\n--{biases}"
        return result

    def initialize_wb(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.random.randn(*np.shape(self.weights[i]))
            self.biases[i] = np.random.randn(*np.shape(self.biases[i]))

    def generate(self):
        for i in range(len(self.layers[1:-1])):
            next_layer = np.matmul(self.weights[i], self.layers[i])
            next_layer = np.add(next_layer, self.biases[i])
            self.weighted_inputs[i+1] = next_layer
            self.layers[i+1] = np.maximum(0, next_layer)
        output_layer = np.matmul(self.weights[-1], self.layers[-2])
        output_layer = np.add(output_layer, self.biases[-1])
        self.weighted_inputs[-1] = output_layer
        self.layers[-1] = bp.softmax(output_layer)

    def normalize(self):
        self.layers[0] /= 255

    def backprop(self, label, layer_index = None):
        self.errors[-1] = bp.output_layer_error(self.layers[-1], label)
        self.weight_gradients[-1] = bp.find_weight_grad(self, -1)
        self.bias_gradients[-1].fill(0)
        self.bias_gradients[-1] = np.add(self.bias_gradients[-1], self.errors)
        for i in range(len(self.errors) - 2, 0, -1):
            self.errors[i] = bp.error_from_next_layer(self, i)
            self.weight_gradients[i] = bp.find_weight_grad(self, i)
            self.bias_gradients[i].fill(0)
            self.bias_gradients[i] = np.add(self.bias_gradients[i], self.errors[i])

    def update(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.subtract(self.weights[i], self.weight_gradients[i] * self.learning_rate)
            self.biases[i] = np.subtract(self.biases[i], self.bias_gradients[i] * self.learning_rate)
    

network = Network(3,5,2, learning_rate=0.01)
network.layers[0] = np.random.randn(*np.shape(network.layers[0]))
network.initialize_wb()
print(network)
network.generate()
print(network)
network.backprop(2)
