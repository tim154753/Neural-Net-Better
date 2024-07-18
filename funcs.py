import numpy as np
import backpropfuncs as bp
import matplotlib.pyplot as plt
import time
#layers size cannot change after network initialization


class Network:
    def __init__(self, *args, learning_rate = 0.1, beta1 = 0.9, beta2 = 0.999, reg_strength = 0.001):
        self.layers = [np.zeros(arg) for arg in args]
        self.weights = [np.zeros((args[i+1], args[i])) for i in range(len(args)-1)]
        self.biases = [np.zeros(arg) for arg in args[1:]]
        self.weighted_inputs = [np.zeros(arg) for arg in args]
        self.errors = [np.zeros(arg) for arg in args[1:]]
        self.weight_gradients = [np.zeros((args[i+1], args[i])) for i in range(len(args)-1)]
        self.bias_gradients = [np.zeros(arg) for arg in args[1:]]
        self.num_layers = len(args)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.reg_strength = reg_strength


    def __str__(self):
        result = "This is the network"
        for i, layer in enumerate(self.layers):
            result += f"\n Layer {i} values: {layer}\n"
        for weights, biases, errors in zip(self.weights, self.biases, self.errors):
            result += f"\nWeights:\n--{weights}\nBiases:\n--{biases}\nErrors:\n{errors}"
        return result

    def initialize_wb(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.random.randn(*np.shape(self.weights[i]))
            self.biases[i] = np.random.randn(*np.shape(self.biases[i]))

    def generate(self):
        for i in range(self.num_layers - 2):
            next_layer = np.matmul(self.weights[i], self.layers[i])
            next_layer = np.add(next_layer, self.biases[i])
            self.weighted_inputs[i+1] = next_layer
            self.layers[i+1] = np.maximum(0, next_layer)  # ReLU
        output_layer = np.matmul(self.weights[-1], self.layers[-2])
        output_layer = np.add(output_layer, self.biases[-1])
        self.weighted_inputs[-1] = output_layer
        self.layers[-1] = bp.softmax(output_layer)

    def backprop(self, label):
        self.errors[len(self.errors) - 1] = bp.output_layer_error(self.layers[-1], label)
        self.weight_gradients[-1] = np.add(self.weight_gradients[-1], bp.find_weight_grad(self, len(self.errors) - 1))
        self.bias_gradients[-1] = np.add(self.bias_gradients[-1], self.errors[-1])
        for i in range(len(self.errors) - 2, -1, -1):
            self.errors[i] = bp.error_from_next_layer(self, i)
            self.weight_gradients[i] = np.add(self.weight_gradients[i], bp.find_weight_grad(self, i))
            self.bias_gradients[i] = np.add(self.bias_gradients[i], self.errors[i])


    def update(self):
        for i in range(len(self.weights)):
            total_weight_gradient = np.add(self.weight_gradients[i],
                                           np.sign(self.weights[i]) * self.reg_strength)
            self.weights[i] = np.subtract(self.weights[i], total_weight_gradient * self.learning_rate)
            self.biases[i] = np.subtract(self.biases[i], self.bias_gradients[i] * self.learning_rate)


    def mini_batch_gradient_descent(self, batch_size, images, labels, test_images, test_labels, num_epochs):
        accuracy_list = []
        velocities = [np.zeros_like(grad) for grad in self.weight_gradients]
        rmsprops = [np.zeros_like(grad) for grad in self.weight_gradients]
        for epoch in range(num_epochs):
            print(f"\nEPOCH: {epoch}\n")
            self.learning_rate = self.learning_rate * pow(1/10, epoch//10)
            indices = np.arange(images.shape[0])
            np.random.shuffle(indices)
            images = images[indices]
            labels = labels[indices]
            for i in range(0, len(images), batch_size):
                self.weight_gradients = [np.zeros_like(grad) for grad in self.weight_gradients]
                batch_images = images[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                for j in range(len(batch_images)):
                    self.layers[0] = batch_images[j]
                    self.generate()
                    self.backprop(batch_labels[j])
                for p in range(len(self.weight_gradients)):
                    self.weight_gradients[p] /= batch_size  # average
                    # velocities[p] = self.beta1 * velocities[p] + (1 - self.beta1) * self.weight_gradients[p]  # momentum
                    # rmsprops[p] = self.beta2 * rmsprops[p] + (1 - self.beta2) * (self.weight_gradients[p] ** 2)  # RMSprop
                    # self.weight_gradients[p] = velocities[p] / ((rmsprops[p] ** 0.5) + 10 ** -8)
                    self.weight_gradients[p] = self.weight_gradients[p] / (abs(self.weight_gradients[p]) + 10 ** -8)
                    self.bias_gradients[p] /= batch_size
                self.update()
            accuracy_list.append(self.test(test_images, test_labels))
            print(f"EPOCH {epoch} accuracy: {accuracy_list[epoch]}\n")
        bp.plot_accuracies(accuracy_list)
        print(f"Accuracy after {num_epochs} epochs: {accuracy_list[-1] * 100}% ")

    def test(self, test_images, label_nums):
        count = 0
        for i in range(10000):
            network.layers[0] = test_images[i]
            network.generate()
            if bp.is_correct(label_nums[i], network.layers[-1]):
                count += 1
        accuracy = count/10000
        return accuracy


network = Network(784, 100, 10, learning_rate=0.1, beta1=0, beta2=0, reg_strength=0.001)
network.initialize_wb()
images = bp.load_all_images()
labels = bp.load_all_labels()
test_images = bp.load_all_test_images()
test_labels = bp.load_all_test_labels()
label_nums = bp.load_all_test_labels("nums")
start = time.time()
network.mini_batch_gradient_descent(100, images, labels, test_images, label_nums, 25)
end = time.time()
print(f"Took {end-start} sec to train")

#  function for user to see network in action by choosing an image from the test set,
#  seeing it, and watching the network classify it
number = int(input("Image number? (type \"-1\" to stop)\n"))
while number != -1:
    if number > 9999:
        number = int(input("Too high. Enter a number from 0-9999\n"))
    network.layers[0] = test_images[number]
    network.generate()
    pixels = test_images[number]
    image = pixels.reshape((28,28))
    plt.imshow(image, cmap='gray')
    plt.show()
    print(np.argmax(network.layers[-1]))
    number = int(input("Image number 0-9999? (type \"-1\" to stop)\n"))
