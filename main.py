import numpy as np
from model import Perceptron
from app import *
# from app import nn, default_model
import copy

#predict images/mnist_png/test/0/10.png
opt = {'predict': 'predict', 'train': 'train', 'train_on_random_images': 'train_on_random_images', \
    'exit' : 'exit', 'load_default_model' : 'load_default_model', 'make_custom_model': 'make_custom_model', \
    'reset_training': 'reset_training', 'load_custom_model': 'load_custom_model', 'save_model': 'save_model'}
while True:
    str = input().split()
    if (str[0] not in opt):
        print(f"you have {len(opt)} mods: {', '.join(opt)}")
        continue
    if (str[0] == opt['exit']):
        break
    if (str[0] == opt['predict']):
        do_predict(str)
    if (str[0] == opt['train']):
        do_train(str)
    if (str[0] == opt["train_on_random_images"]):
        do_train_on_random_images(str)
    if (str[0] == opt['load_default_model']):
        do_load_default_model()

    if (str[0] == opt["make_custom_model"]):
        do_make_custom_model(str)
    if (str[0] == opt["reset_training"]):
        nn = copy.copy(default_model)
    if (str[0] == 'load_custom_model'):
        do_load_custom_model(str)
    if (str[0] == 'save_model'):
        do_save_model(str)
