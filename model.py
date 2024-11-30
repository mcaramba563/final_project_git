import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import os
from utils import file_exists

class Perceptron:    
    def gen_weights(self):
        self.weights_input_hidden = []
        self.weights_input_hidden.append(np.random.rand(self.input_size, self.hidden_layers_size[0]) - 0.5)
        for j in range(1, len(self.hidden_layers_size)):
            self.weights_input_hidden.append(np.random.rand(self.hidden_layers_size[j - 1], self.hidden_layers_size[j]) - 0.5)
        self.weights_output_hidden = np.random.rand(self.hidden_layers_size[-1], self.output_size) - 0.5
        self.weights_output_hidden = np.asarray(self.weights_output_hidden, dtype=np.float64)

    def bias_gen(self):
        self.bias_input_hidden = [None for _ in range(len(self.hidden_layers_size))]
        for j in range(0, len(self.hidden_layers_size)):
            self.bias_input_hidden[j] = np.random.rand(1, self.hidden_layers_size[j]) - 0.5
    
    def __init__(self, input_size=28*28, hidden_layers_size=[400, 256, 128], output_size=10, learning_rate=0.01, epochs=1, path_to_load=None):
        
            
        self.input_size = input_size
        self.hidden_layers_size = hidden_layers_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        np.random.seed(42) 
        self.gen_weights()
        self.bias_gen()

        if (path_to_load is not None):
                self.load_model(path_to_load)
    
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, X):
        hidden_layer_input = [None for _ in range(len(self.weights_input_hidden))]
        hidden_layer_output = [None for _ in range(len(self.weights_input_hidden))]
        for j in range(len(self.weights_input_hidden)):
            if (j == 0):
                hidden_layer_input[j] = np.dot(X, self.weights_input_hidden[0]) + self.bias_input_hidden[j]
            else:
                hidden_layer_input[j] = np.dot(hidden_layer_output[j - 1], self.weights_input_hidden[j]) + self.bias_input_hidden[j]
                #print(hidden_layer_input[j].shape)
                #break
            hidden_layer_output[j] = self.tanh(hidden_layer_input[j])
        #break
        final_hidden_layer_input = np.dot(hidden_layer_output[-1], self.weights_output_hidden) 
        final_hidden_layer_output = self.tanh(final_hidden_layer_input)

        final_output = self.softmax(final_hidden_layer_output)
        self.hidden_layer_output = hidden_layer_output
        self.final_hidden_layer_output = final_hidden_layer_output
        self.X = X
        return final_output

    def backprop(self, y_train_one_hot):
        X = self.X
        hidden_layer_output = self.hidden_layer_output
        final_output = self.final_hidden_layer_output
        error_final = y_train_one_hot - final_output
        
        d_final_output = error_final * self.tanh_derivative(final_output)
        d_hidden_layers = [None for _ in range(len(self.weights_input_hidden))]
    
        for j in reversed(range(len(self.weights_input_hidden))):
            if (j == len(self.weights_input_hidden) - 1):
                cur_error = np.dot(d_final_output, self.weights_output_hidden.T)
                d_hidden_layers[j] = cur_error * self.tanh_derivative(hidden_layer_output[j])
            else:
                cur_error = np.dot(d_hidden_layers[j + 1], self.weights_input_hidden[j + 1].T)
                d_hidden_layers[j] = cur_error * self.tanh_derivative(hidden_layer_output[j])
                #print(cur_error.shape, d_hidden_layers[j].shape)
                #break
        #break
        #weights update
        self.weights_output_hidden += hidden_layer_output[-1].reshape(-1, 1).dot(d_final_output.reshape(1, -1)) * self.learning_rate
        
        
        for j in range(len(self.weights_input_hidden)):
            if (j > 0):
                self.weights_input_hidden[j] += hidden_layer_output[j - 1].reshape(-1, 1).dot(d_hidden_layers[j].reshape(1, -1)) * self.learning_rate
                self.bias_input_hidden[j] += d_hidden_layers[j] * self.learning_rate
            else:
                self.weights_input_hidden[j] += X.reshape(-1, 1).dot(d_hidden_layers[j].reshape(1, -1)) * self.learning_rate
                self.bias_input_hidden[j] += d_hidden_layers[j] * self.learning_rate
        pass

    def train(self, X_train, y_train, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        y_train_one_hot = np.zeros((len(y_train), self.output_size))
        #print(X_train, y_train)
        for i in range(len(y_train)):
            y_train_one_hot[i][int(y_train[i])] = 1
        for epoch in range(self.epochs):
            for ind in range(len(X_train)):
                if (ind % 10000 == 0 and ind > 0):
                    print(ind)
                
                self.forward(X_train[ind].reshape(-1))
                self.backprop(y_train_one_hot[ind])
            print(f'Epoch {epoch + 1}/{self.epochs} completed')
        pass

    def train_on_images(self, pathes, y_train, epochs, learning_rate):
        X = []
        for cur in pathes:
            im = iio.imread(cur)
            X.append(im)
        X = np.array(X)
        self.train(X, y_train, epochs, learning_rate)
    
    def predict(self, X):
        predicted_output = self.forward(X.reshape(-1))
        return np.argmax(predicted_output)
        
    def predict_image(self, path):
        if (file_exists(path) == False):
            return -1
        #predict images/mnist_png/test/0/10.png
        im = iio.imread(path)
        return self.predict(im)
    def save_model(self, path):
        save_array = np.array([np.asarray(self.weights_input_hidden, dtype="object"), self.weights_output_hidden, np.asarray([cur.reshape(-1) for cur in self.bias_input_hidden], dtype="object")], dtype="object")
        if ('/' not in path):
            np.save(path, save_array, allow_pickle=True)
            return
        fold = '/'.join(path.split('/')[:-1])
        os.makedirs(fold, exist_ok=True)
        np.save(path, save_array, allow_pickle=True)
        
    def load_model(self, path):
        save_array = np.load(path, allow_pickle=True)
        self.weights_input_hidden = save_array[0]
        self.weights_output_hidden = save_array[1]
        self.bias_input_hidden = np.asarray([cur for cur in save_array[2]], dtype="object")