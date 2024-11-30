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