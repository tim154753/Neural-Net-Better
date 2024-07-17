import numpy as np
import backpropfuncs as bp
import matplotlib.pyplot as plt
import time
import tkinter as tk
from PIL import Image, ImageDraw
#layers size cannot change after network initialization


class Network:
    def __init__(self, *args, learning_rate = 0.1, reg_strength = 0.01):
        self.layers = [np.zeros(arg) for arg in args]
        self.weights = [np.zeros((args[i+1], args[i])) for i in range(len(args)-1)]
        self.biases = [np.zeros(arg) for arg in args[1:]]
        self.weighted_inputs = [np.zeros(arg) for arg in args]
        self.errors = [np.zeros(arg) for arg in args[1:]]
        self.weight_gradients = [np.zeros((args[i+1], args[i])) for i in range(len(args)-1)]
        self.bias_gradients = [np.zeros(arg) for arg in args[1:]]
        self.num_layers = len(args)
        self.learning_rate = learning_rate
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
        #print(f"From backprop: this is error[-1]:\n{self.errors[-1]}")
        self.weight_gradients[-1] = np.add(self.weight_gradients[-1], bp.find_weight_grad(self, len(self.errors) - 1))
        #self.bias_gradients[-1].fill(0)
        self.bias_gradients[-1] = np.add(self.bias_gradients[-1], self.errors[-1])
        for i in range(len(self.errors) - 2, -1, -1):
            #print(f"From backprop: this is weights for layer {i} (shape = {np.shape(self.weights[i])}:\n{self.weights[i]}")
            self.errors[i] = bp.error_from_next_layer(self, i)
            #print(f"From backprop: this is error {i}:\n{self.errors[i]}")
            self.weight_gradients[i] = np.add(self.weight_gradients[i], bp.find_weight_grad(self, i))
            #self.bias_gradients[i].fill(0)
            self.bias_gradients[i] = np.add(self.bias_gradients[i], self.errors[i])


    def update(self):
        for i in range(len(self.weights)):
            total_weight_gradient = np.add(self.weight_gradients[i] * self.learning_rate,
                                           np.sign(self.weights[i]) * self.reg_strength)
            self.weights[i] = np.subtract(self.weights[i], total_weight_gradient)
            self.biases[i] = np.subtract(self.biases[i], self.bias_gradients[i] * self.learning_rate)


    def mini_batch_gradient_descent(self, batch_size, images, labels, test_images, test_labels, num_epochs):
        accuracy_list = []
        for epoch in range(num_epochs):
            print(f"\nEPOCH: {epoch}\n")
            indices = np.arange(images.shape[0])
            np.random.shuffle(indices)
            images = images[indices]
            labels = labels[indices]
            # for k in range(len(self.weight_gradients)):
                # self.weight_gradients[k].fill(0)
                # self.bias_gradients[k].fill(0)
            for i in range(0, len(images), batch_size):
                #print(f"On batch {i}!\n")
                batch_images = images[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                for j in range(len(batch_images)):
                    self.layers[0] = batch_images[j]
                    self.generate()
                    self.backprop(batch_labels[j])
                for p in range(len(self.weight_gradients)):
                    self.weight_gradients[p] /= batch_size
                    self.bias_gradients[p] /= batch_size
                self.update()
            accuracy_list.append(self.test(test_images, test_labels))
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

network = Network(784, 100, 10, learning_rate=0.01)
network.initialize_wb()
print(network)

images = bp.load_all_images()
labels = bp.load_all_labels()
test_images = bp.load_all_test_images()
test_labels = bp.load_all_test_labels()
label_nums = bp.load_all_test_labels("nums")
start = time.time()
network.mini_batch_gradient_descent(100, images, labels, test_images, label_nums, 10)
end = time.time()
print(f"Took {end-start} sec to train")

network.test(test_images, label_nums)


number = int(input("Image number? (type \"-1\" to stop)\n"))
while number != -1:
    network.layers[0] = images[number]
    network.generate()
    pixels = images[number]
    image = pixels.reshape((28,28))
    plt.imshow(image, cmap='gray')
    plt.show()
    print(np.argmax(network.layers[-1]))
    number = int(input("Image number? (type \"-1\" to stop)\n"))

'''
window = tk.Tk()
canvas = tk.Canvas(window, width=500, height=500)
canvas.pack()
image = Image.new("L", (500, 500), "black")
draw = ImageDraw.Draw(image)

def draw_on_canvas(event):
    radius = 10
    x1, y1 = (event.x - radius), (event.y - radius)
    x2, y2 = (event.x + radius), (event.y + radius)
    canvas.create_oval(x1, y1, x2, y2, fill='white')
    # draw.line([x1, y1, x2, y2], fill="black", width=50)
    draw.ellipse([x1, y1, x2, y2], fill="white")

canvas.bind("<B1-Motion>", draw_on_canvas)

def classify_image():
    resized_image = image.resize((28, 28)).convert("L")
    image_pixels = np.asarray(resized_image) / 255.0
    image_pixels[image_pixels < 0.1] = 0
    plt.imshow(image_pixels * 255, cmap='gray')
    plt.show()
    image_pixels = image_pixels.reshape(784, )
    # print(image_pixels)
    network.layers[0] = image_pixels
    network.generate()
    print(np.argmax(network.layers[-1]))

def clear_image():
    canvas.delete("all")
    global image, draw
    image = Image.new("RGB", (500, 500), "black")
    draw = ImageDraw.Draw(image)

classify_button = tk.Button(window, text="Classify", command=classify_image)
classify_button.pack()

clear_button = tk.Button(window, text="Clear", command=clear_image)
clear_button.pack()

window.mainloop()
'''
