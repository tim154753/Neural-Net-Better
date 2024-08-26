import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import backpropfuncs as bp
import matplotlib.pyplot as plt
import time

  # forces numpy to run on a single thread (hopefully)
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
            total_weight_gradient = np.add(self.weight_gradients[i], np.sign(self.weights[i]) * self.reg_strength)
            self.weights[i] = np.subtract(self.weights[i], total_weight_gradient * self.learning_rate)
            self.biases[i] = np.subtract(self.biases[i], self.bias_gradients[i] * self.learning_rate)


    def mini_batch_gradient_descent(self, batch_size, images, labels, test_images, test_labels, num_epochs):
        accuracy_list = []
        velocities = [np.zeros_like(grad) for grad in self.weight_gradients]
        rmsprops = [np.zeros_like(grad) for grad in self.weight_gradients]
        for epoch in range(num_epochs):
            if epoch >= 3:
                if abs(2 * accuracy_list[-1] - (accuracy_list[-2] + accuracy_list[-3])) < 10 ** -4:
                    if input("Training seems to have plateaued. Do you wish to continue?\n") == "no":
                        break
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
                    #self.layers[0] = batch_images[j]
                    self.layers[0] = self.image_randomize(batch_images[j])  # randomly flip input image
                    self.generate()
                    self.backprop(batch_labels[j])
                for p in range(len(self.weight_gradients)):
                    self.weight_gradients[p] /= batch_size  # average
                    #velocities[p] = self.beta1 * velocities[p] + (1 - self.beta1) * self.weight_gradients[p]  # momentum
                    #rmsprops[p] = self.beta2 * rmsprops[p] + (1 - self.beta2) * (self.weight_gradients[p] ** 2)  # RMSprop
                    #self.weight_gradients[p] = velocities[p] / ((rmsprops[p] ** 0.5) + 10 ** -8)
                    #self.weight_gradients[p] = self.weight_gradients[p] / (abs(self.weight_gradients[p]) + 10 ** -8)
                    # above line normalizes weight gradient to either 0 or 1, immensely helps training for some reason
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

    def image_randomize(self, image):
        rand = np.random.random()
        image = image.reshape(28,28)
        if rand <= 0.33:
            vert_flip = np.flipud(image)
            return vert_flip.reshape(784)
        if rand > 0.33 and rand <=0.66:
            hor_flip = np.fliplr(image)
            return hor_flip.reshape(784)
        return image.reshape(784)
    def saliency_map(self, image, label = None):
        self.layers[0] = image
        self.generate()
        # network now generates output based on image
        # 3 different activation functions -> 3 different stages of backwards propagation here
        # first step = from output layer (act func = softmax)
        # grad is a tensor of dim. j x j x k, must sum across horizontal j ("derivatives per neuron dimension")
        # generate output grad
        prime = bp.softmax_prime(self.layers[-1])
        j, k = self.weights[-1].shape
        output_grad = np.zeros((k,j,j))
        for col_num in range(k):
            weights = self.weights[-1]
            col = weights[:, col_num]  # gets one column of weight matrix
            output_grad[col_num,:,:] = prime * col  # uses broadcasting to assign each layer
            # of output_grad with the penetrating face product of the Jacobian of softmax and the weight column
        output_grad = np.sum(output_grad, axis=2)
        # summing along dimension 2 of the tensor
        output_grad = np.transpose(output_grad)
        if label is not None:
            loss_grad = np.zeros(self.layers[-1].shape)
            loss_grad[label] = -1/self.layers[-1][label]
            output_grad = np.matmul(loss_grad, output_grad)
        # after all that, output_grad should be of shape jxk
        # now backprop through hidden layers (ReLU act. func)
        current_step = output_grad
        for i in range(len(self.layers) - 2, 0, -1):
            grad = self.weights[i-1] * bp.relu_prime(self.layers[i]).reshape(self.layers[i].size, 1)
            current_step = np.matmul(current_step, grad)
        return current_step  #  gradient of output activations with respect to input activations, of
        # shape (output_layer.size, input_layer.size) -> in this case (10, 784)


network = Network(784, 100, 50, 10, learning_rate=0.1, beta1=0.9, beta2=0.99, reg_strength=0)
network.initialize_wb()
images = bp.load_all_images()
labels = bp.load_all_labels()
test_images = bp.load_all_test_images()
test_labels = bp.load_all_test_labels()
label_nums = bp.load_all_test_labels("nums")
start = time.time()
network.mini_batch_gradient_descent(100, images, labels, test_images, label_nums, 100)
end = time.time()

print(f"Took {end-start} sec to train")

#  saliency map with respect to loss function

test_image = test_images[1000]
test_label = label_nums[1000]
network.layers[0] = test_image
network.generate()
print(np.argmax(network.layers[-1]))
loss_map = network.saliency_map(test_image, test_label)
loss_pixels = loss_map.reshape(28,28)
# normalize
loss_pixels = loss_pixels / (np.max(loss_pixels) + 10**-6)
test_display = test_image.reshape(28,28)
fig, ax = plt.subplots()
ax.imshow(test_display, cmap='gray')
ax.imshow(loss_pixels, cmap='jet', alpha=0.5)
plt.show()


#  function for user to see network in action by choosing an image from the test set,
#  seeing it, and watching the network classify it
number = int(input("Image number? (type \"-1\" to stop)\n"))
while number != -1:
    if number > 9999:
        number = int(input("Too high. Enter a number from 0-9999\n"))
    network.layers[0] = test_images[number]
    network.generate()
    guess = np.argmax(network.layers[-1])
    pixels = test_images[number]
    image = pixels.reshape((28,28))
    #plt.imshow(image, cmap='gray')
    #plt.show()
    map = network.saliency_map(test_images[number])
    map_pix = map[guess]  # shows what network "was looking at" to make this guess
    map_img = map_pix.reshape((28,28))
    map_img = (map_img) / ((np.max(map_img)) + 10**-6) # normalization

    # overlay map as heatmap on top of input image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.imshow(map_img, cmap='jet', alpha=0.5)
    plt.show()

    print(guess)
    number = int(input("Image number 0-9999? (type \"-1\" to stop)\n"))

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
