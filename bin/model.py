import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import os
from bin.utils import file_exists

class Perceptron:    

    def gen_weights(self):
        """
        Generates random weights for the input-hidden and hidden-output layers.
        """
        self.weights_input_hidden = []
        """A list of weight matrices connecting input to hidden layers and between hidden layers."""
        self.weights_input_hidden.append(np.random.rand(self.input_size, self.hidden_layers_size[0]) - 0.5)
        for j in range(1, len(self.hidden_layers_size)):
            self.weights_input_hidden.append(np.random.rand(self.hidden_layers_size[j - 1], self.hidden_layers_size[j]) - 0.5)
        self.weights_output_hidden = np.random.rand(self.hidden_layers_size[-1], self.output_size) - 0.5
        """The weight matrix connecting the last hidden layer to the output layer."""

    def bias_gen(self):
        """
        Generates random biases for the hidden layers.
        """
        self.bias_input_hidden = [None for _ in range(len(self.hidden_layers_size))]
        """A list of bias vectors for each hidden layer."""
        for j in range(0, len(self.hidden_layers_size)):
            self.bias_input_hidden[j] = np.random.rand(1, self.hidden_layers_size[j]) - 0.5
    
    def __init__(self, input_size=28*28, hidden_layers_size=[400, 256, 128], output_size=10, learning_rate=0.01, epochs=1, path_to_load=None):
        """
        Initializes the Perceptron model with the given architecture and parameters.

        :param input_size: Size of the input layer
        :type input_size: int
        :param hidden_layers_size: List of sizes of hidden layers
        :type hidden_layers_size: list of int
        :param output_size: Size of the output layer
        :type output_size: int
        :param learning_rate: Learning rate for weight updates
        :type learning_rate: float
        :param epochs: Number of training epochs
        :type epochs: int
        :param path_to_load: Path to load pre-trained model (optional)
        :type path_to_load: str
        """ 
        self.input_size = input_size
        """The size of the input layer, representing the number of input features."""
        self.hidden_layers_size = hidden_layers_size
        """A list where each element specifies the number of neurons in a hidden layer."""
        self.output_size = output_size
        """The size of the output layer, representing the number of output classes."""
        self.learning_rate = learning_rate
        """The learning rate used for updating weights and biases during training."""
        self.epochs = epochs
        """The number of epochs to train the model."""
        
        np.random.seed(42) 
        self.gen_weights()
        self.bias_gen()

        if (path_to_load is not None):
            self.load_model(path_to_load)
    
    def tanh(self, x):
        """
        Applies the tanh activation function to the input.

        :param x: Input array
        :type x: numpy.ndarray
        :return: Transformed array
        :rtype: numpy.ndarray
        """
        return np.tanh(x)

    def tanh_derivative(self, x):
        """
        Computes the derivative of the tanh activation function.

        :param x: Input array
        :type x: numpy.ndarray
        :return: Derivative of tanh
        :rtype: numpy.ndarray
        """
        return 1 - np.tanh(x) ** 2

    def softmax(self, x):
        """
        Applies the softmax function to the input.

        :param x: Input array
        :type x: numpy.ndarray
        :return: Transformed array
        :rtype: numpy.ndarray
        """
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, X):
        """
        Performs a forward pass through the network.

        :param X: Input data
        :type X: numpy.ndarray
        :return: Network output
        :rtype: numpy.ndarray
        """
        hidden_layer_input = [None for _ in range(len(self.weights_input_hidden))]
        hidden_layer_output = [None for _ in range(len(self.weights_input_hidden))]
        for j in range(len(self.weights_input_hidden)):
            if (j == 0):
                hidden_layer_input[j] = np.dot(X, self.weights_input_hidden[0]) + self.bias_input_hidden[j]
            else:
                hidden_layer_input[j] = np.dot(hidden_layer_output[j - 1], self.weights_input_hidden[j]) + self.bias_input_hidden[j]
            hidden_layer_output[j] = self.tanh(hidden_layer_input[j])

        final_hidden_layer_input = np.dot(hidden_layer_output[-1], self.weights_output_hidden) 
        final_hidden_layer_output = self.tanh(final_hidden_layer_input)

        final_output = self.softmax(final_hidden_layer_output)
        self.hidden_layer_output = hidden_layer_output
        """Stores the output of each hidden layer during forward propagation."""
        self.final_hidden_layer_output = final_hidden_layer_output
        """The output of the last hidden layer during forward propagation."""
        self.X = X
        """Stores the input data passed through the model during the current forward pass."""
        return final_output

    def backprop(self, y_train_one_hot):
        """
        Performs backpropagation to update weights and biases.

        :param y_train_one_hot: One-hot encoded labels
        :type y_train_one_hot: numpy.ndarray
        """
        X = self.X
        hidden_layer_output = self.hidden_layer_output
        final_output = self.final_hidden_layer_output
        error_final = y_train_one_hot - final_output
        
        d_final_input = error_final * self.tanh_derivative(final_output) # before tanh
        d_hidden_layers = [None for _ in range(len(self.weights_input_hidden))]
    
        for j in reversed(range(len(self.weights_input_hidden))):
            if (j == len(self.weights_input_hidden) - 1):
                cur_error = np.dot(d_final_input, self.weights_output_hidden.T)
                d_hidden_layers[j] = cur_error * self.tanh_derivative(hidden_layer_output[j])
            else:
                cur_error = np.dot(d_hidden_layers[j + 1], self.weights_input_hidden[j + 1].T)
                d_hidden_layers[j] = cur_error * self.tanh_derivative(hidden_layer_output[j])
                
        self.weights_output_hidden += hidden_layer_output[-1].reshape(-1, 1).dot(d_final_input.reshape(1, -1)) * self.learning_rate
        
        
        for j in range(len(self.weights_input_hidden)):
            if (j > 0):
                self.weights_input_hidden[j] += hidden_layer_output[j - 1].reshape(-1, 1).dot(d_hidden_layers[j].reshape(1, -1)) * self.learning_rate
                self.bias_input_hidden[j] += d_hidden_layers[j] * self.learning_rate
            else:
                self.weights_input_hidden[j] += X.reshape(-1, 1).dot(d_hidden_layers[j].reshape(1, -1)) * self.learning_rate
                self.bias_input_hidden[j] += d_hidden_layers[j] * self.learning_rate
        pass

    def train(self, X_train, y_train, epochs, learning_rate):
        """
        Trains the perceptron model on the provided data.

        :param X_train: Training input data
        :type X_train: numpy.ndarray
        :param y_train: Training labels
        :type y_train: numpy.ndarray
        :param epochs: Number of epochs to train
        :type epochs: int
        :param learning_rate: Learning rate for training
        :type learning_rate: float
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        y_train_one_hot = np.zeros((len(y_train), self.output_size))
        
        for i in range(len(y_train)):
            y_train_one_hot[i][int(y_train[i])] = 1
        for epoch in range(self.epochs):
            for ind in range(len(X_train)):
                self.forward(X_train[ind].reshape(-1))
                self.backprop(y_train_one_hot[ind])
            print(f'Epoch {epoch + 1}/{self.epochs} completed')
        pass

    def train_on_images(self, pathes, y_train, epochs, learning_rate):
        """
        Trains the perceptron model using image data from the specified paths.

        :param pathes: List of file paths to training images
        :type pathes: list of str
        :param y_train: Training labels
        :type y_train: numpy.ndarray
        :param epochs: Number of epochs to train
        :type epochs: int
        :param learning_rate: Learning rate for training
        :type learning_rate: float
        """
        X = []
        for cur in pathes:
            im = iio.imread(cur)
            X.append(im)
        X = np.array(X)
        self.train(X, y_train, epochs, learning_rate)
    
    def predict(self, X):
        """
        Predicts the output class for the given input data.

        :param X: Input data
        :type X: numpy.ndarray
        :returns: Predicted class index
        :rtype: int
        """
        predicted_output = self.forward(X.reshape(-1))
        return np.argmax(predicted_output)
        
    def predict_image(self, path):
        """
        Predicts the class of an image at the specified path.

        :param path: Path to the image file
        :type path: str
        :returns: Predicted class index, or -1 if the file doesn't exist
        :rtype: int
        """
        if (file_exists(path) == False):
            return -1
        
        im = iio.imread(path)
        return self.predict(im)
    def save_model(self, path):
        """
        Saves the model's weights and biases to a file.

        :param path: File path to save the model
        :type path: str
        """
        save_array = np.array([np.asarray(self.weights_input_hidden, dtype="object"), self.weights_output_hidden, np.asarray([cur.reshape(-1) for cur in self.bias_input_hidden], dtype="object")], dtype="object")
        if ('/' not in path):
            np.save(path, save_array, allow_pickle=True)
            return
        fold = '/'.join(path.split('/')[:-1])
        os.makedirs(fold, exist_ok=True)
        np.save(path, save_array, allow_pickle=True)
        
    def load_model(self, path):
        """
        Loads model weights and biases from a file.

        :param path: File path to load the model from
        :type path: str
        """
        save_array = np.load(path, allow_pickle=True)
        self.weights_input_hidden = save_array[0]
        self.weights_output_hidden = save_array[1]
        self.bias_input_hidden = np.asarray([cur for cur in save_array[2]], dtype="object")