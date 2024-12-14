import numpy as np
from bin.model import Perceptron
from bin.utils import get_random_pathes
from bin.utils import file_exists
import copy
from bin.app import *

class App:
    def __init__(self):
        self.default_model = Perceptron(path_to_load='models/folder2/model2.npy')
        """A default perceptron model loaded from a predefined file."""
        self.nn = Perceptron(path_to_load='models/folder2/model2.npy')
        """The current perceptron model in use."""

    def do_predict(self, str):
        """
            Predicts the class of an image given its file path.

            :param str: List of strings where the second element is the image file path
            :type str: list of str
            :returns: 0 on success, -1 on failure
            :rtype: int
        """
        try:
            if len(str) < 2:
                print("You need to specify the path to file")
                return -1
            ans = self.nn.predict_image(str[1])
            if ans == -1:
                print('Something went wrong. Check the path to the file and try again.')
                return -1
            print(ans)
            return 0
        except Exception as e:
            print(f"Error in prediction: {e}")
            return -1

    def do_train(self, str):
        """
            Trains the model on labeled image data from a file.

            :param str: List of strings containing the file path, epochs, and learning rate
            :type str: list of str
            :returns: 0 on success, -1 on failure
            :rtype: int
        """
        try:
            if len(str) < 4:
                print("You need to specify the path to file (with names of files and labels), epochs, and learning rate")
                return -1

            epochs = int(str[2])
            learning_rate = float(str[3])
            if epochs <= 0 or learning_rate <= 0:
                print('Check epochs and learning_rate values')
                return -1

            pathes, labels = [], []
            name_of_file = str[1]
            if not file_exists(name_of_file):
                print('Check the name of the file')
                return -1
            with open(name_of_file, 'r') as f:
                data = f.read()
                for cur_str in data.split("\n"):
                    if not cur_str.strip():
                        continue
                    try:
                        path, label = tuple(cur_str.split())
                        if not file_exists(path):
                            print(f'Path {path} is not valid')
                            return -1
                        pathes.append(path)
                        labels.append(int(label))
                    except ValueError:
                        print(f"Invalid format in file: {cur_str}")
                        return -1

            self.nn.train_on_images(pathes, labels, epochs, learning_rate)
            return 0

        except Exception as e:
            print(f"Error during training: {e}")
            return -1


    def do_train_on_random_images(self, str):
        """
            Trains the model on a random selection of labeled images.

            :param str: List of strings containing the number of images, epochs, and learning rate
            :type str: list of str
            :returns: 0 on success, -1 on failure
            :rtype: int
        """
        try:
            if len(str) < 4:
                print('You need to specify count of images, epochs, and learning_rate')
                return -1

            n = int(str[1])
            epochs = int(str[2])
            learning_rate = float(str[3])
            if n <= 0 or epochs <= 0 or learning_rate <= 0:
                print('Check count of images, epochs, and learning_rate values.')
                return -1

            pathes, labels = get_random_pathes(n)
            self.nn.train_on_images(pathes, labels, epochs, learning_rate)
            return 0
        except Exception as e:
            print(f"Error during training on random images: {e}")
            return -1

    def do_make_custom_model(self, str):
        """
            Creates a custom perceptron model with specified hidden layer sizes.

            :param str: List of strings where each element represents the number of neurons in a hidden layer
            :type str: list of str
            :returns: 0 on success, -1 on failure
            :rtype: int
        """
        try:
            if len(str) < 2:
                print('You need to specify the count of neurons in each layer')
                return -1

            hidden_layers_size = [int(cur) for cur in str[1:]]
            self.nn = Perceptron(input_size=28*28, hidden_layers_size=hidden_layers_size, output_size=10, learning_rate=0.01, epochs=1)
            self.default_model = copy.copy(self.nn)
            return 0
        except ValueError as e:
            print(f"Invalid neuron count: {e}")
            return -1
        except Exception as e:
            print(f"Error creating custom model: {e}")
            return -1

    def do_load_custom_model(self, str):
        """
            Loads a custom perceptron model from a specified file path.

            :param str: List of strings where the second element is the model file path
            :type str: list of str
            :returns: 0 on success, -1 on failure
            :rtype: int
        """
        try:
            if len(str) < 2:
                print('You need to specify the path to the file.')
                return -1
            if not file_exists(str[1]):
                print('Invalid file name')
                return -1
            
            self.nn = Perceptron(path_to_load=str[1])
            self.default_model = copy.copy(self.nn)
            return 0
        except Exception as e:
            print(f"Error loading custom model: {e}")
            return -1


    def do_save_model(self, str):
        """
            Saves the current model to a specified file path.

            :param str: List of strings where the second element is the save file path
            :type str: list of str
            :returns: 0 on success, -1 on failure
            :rtype: int
        """
        try:
            if len(str) < 2:
                print('Specify the path to the file')
                return -1
            self.nn.save_model(path=str[1])
            return 0
        except PermissionError:
            print("Permission denied. Check file path and access rights.")
            return -1
        except Exception as e:
            print(f"Error saving model: {e}")
            return -1

    def do_load_default_model(self):
        """
            Loads the default perceptron model from a predefined file.
        """
        
        self.nn = Perceptron(path_to_load='models/folder2/model2.npy')
        self.default_model = copy.copy(self.nn)

    def reset_training(self):
        """
            Reset training
        """
        self.nn = copy.copy(self.default_model)