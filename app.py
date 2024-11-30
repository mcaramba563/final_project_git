import numpy as np
from model import Perceptron
from utils import get_random_pathes
from utils import file_exists
import copy
from app import *

default_model = Perceptron(path_to_load='models/folder2/asd.npy')
nn = Perceptron(path_to_load='models/folder2/asd.npy')

def do_predict(str):
    try:
        if len(str) < 2:
            print("You need to specify the path to file")
            return -1
        ans = nn.predict_image(str[1])
        if ans == -1:
            print('Something went wrong. Check the path to the file and try again.')
            return -1
        print(ans)
        return 0
    except Exception as e:
        print(f"Error in prediction: {e}")
        return -1

def do_train(str):
    try:
        if len(str) < 4:
            print("You need to specify the path to file (with names of files and labels), epochs, and learning rate")
            return -1

        global nn
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

        nn.train_on_images(pathes, labels, epochs, learning_rate)
        return 0

    except Exception as e:
        print(f"Error during training: {e}")
        return -1


def do_train_on_random_images(str):
    try:
        if len(str) < 4:
            print('You need to specify count of images, epochs, and learning_rate')
            return -1

        global nn
        n = int(str[1])
        epochs = int(str[2])
        learning_rate = float(str[3])
        if n <= 0 or epochs <= 0 or learning_rate <= 0:
            print('Check count of images, epochs, and learning_rate values.')
            return -1

        pathes, labels = get_random_pathes(n)
        nn.train_on_images(pathes, labels, epochs, learning_rate)
        return 0
    except Exception as e:
        print(f"Error during training on random images: {e}")
        return -1

def do_make_custom_model(str):
    try:
        if len(str) < 2:
            print('You need to specify the count of neurons in each layer')
            return -1

        global nn, default_model
        hidden_layers_size = [int(cur) for cur in str[1:]]
        nn = Perceptron(input_size=28*28, hidden_layers_size=hidden_layers_size, output_size=10, learning_rate=0.01, epochs=1)
        default_model = copy.copy(nn)
        return 0
    except ValueError as e:
        print(f"Invalid neuron count: {e}")
        return -1
    except Exception as e:
        print(f"Error creating custom model: {e}")
        return -1

def do_load_custom_model(str):
    try:
        if len(str) < 2:
            print('You need to specify the path to the file.')
            return -1
        if not file_exists(str[1]):
            print('Invalid file name')
            return -1
        global nn
        global default_model
        nn = Perceptron(path_to_load=str[1])
        default_model = copy.copy(nn)
        return 0
    except Exception as e:
        print(f"Error loading custom model: {e}")
        return -1


def do_save_model(str):
    try:
        if len(str) < 2:
            print('Specify the path to the file')
            return -1
        nn.save_model(path=str[1])
        return 0
    except PermissionError:
        print("Permission denied. Check file path and access rights.")
        return -1
    except Exception as e:
        print(f"Error saving model: {e}")
        return -1

def do_load_default_model():
    global nn
    global default_model
    nn = Perceptron(path_to_load='models/folder2/asd.npy')
    default_model = copy.copy(nn)