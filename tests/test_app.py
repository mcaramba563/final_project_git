import pytest
from unittest.mock import Mock
import builtins

import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bin.app import *

@pytest.fixture
def app_instance():
    return App()

def count_of_right(test_cases, app_instance):
    mock_print = Mock()
    builtins.print = mock_print
    cnt_right = 0
    for command, expected in test_cases:
        app_instance.do_predict(command)
        actual = int(mock_print.call_args[0][0])
        if actual == expected:
            cnt_right += 1

    return cnt_right

def generate_standart_test_cases():
    return [
        ('predict images/mnist_png/test/0/10.png'.split(), 0),
        ('predict images/mnist_png/test/1/2.png'.split(), 1),
        ('predict images/mnist_png/test/2/1.png'.split(), 2),
        ('predict images/mnist_png/test/3/18.png'.split(), 3),
        ('predict images/mnist_png/test/4/4.png'.split(), 4),
        ('predict images/mnist_png/test/5/15.png'.split(), 5),
        ('predict images/mnist_png/test/6/11.png'.split(), 6),
        ('predict images/mnist_png/test/7/0.png'.split(), 7),
        ('predict images/mnist_png/test/8/61.png'.split(), 8),
        ('predict images/mnist_png/test/9/7.png'.split(), 9),
    ]

def test_valid_predict(app_instance):
    test_cases = generate_standart_test_cases()

    cnt_right = count_of_right(test_cases, app_instance)

    assert cnt_right == 10

def test_incorrect_predict(app_instance):
    assert app_instance.do_predict('predict images/mnist_png/test/9asdad/7.png'.split()) == -1
    assert app_instance.do_predict('predict'.split()) == -1

def test_incorrect_train(app_instance):
    assert app_instance.do_train('train tests/file_with_names.txt'.split()) == -1
    assert app_instance.do_train('train tests/file_with_names.txt 2'.split()) == -1
    assert app_instance.do_train('train tests/file_with_names.txt -1 0.01'.split()) == -1
    assert app_instance.do_train('train asd.txt 1 0.01'.split()) == -1
    assert app_instance.do_train('train tests/file_with_incorrects_names.txt 1 0.01'.split()) == -1

def test_correct_train(app_instance):
    assert app_instance.do_train('train tests/file_with_names.txt 1 0.01'.split()) == 0

def test_train_on_random_images(app_instance):
    assert app_instance.do_train_on_random_images('train_on_random_images 100'.split()) == -1
    assert app_instance.do_train_on_random_images('train_on_random_images 100 1'.split()) == -1
    assert app_instance.do_train_on_random_images('train_on_random_images 100 0 0.01'.split()) == -1
    assert app_instance.do_train_on_random_images('train_on_random_images 0 10 0.01'.split()) == -1

    assert app_instance.do_train_on_random_images('train_on_random_images 10 10 0.01'.split()) == 0

def test_make_custom_model(app_instance):
    assert app_instance.do_make_custom_model('make_custom_model 400 256 128'.split()) == 0
    assert app_instance.do_train_on_random_images('train_on_random_images 20000 2 0.01'.split()) == 0

    test_cases = generate_standart_test_cases()
    assert count_of_right(test_cases, app_instance) >= 7


def load_custom_model(app_instance):
    assert app_instance.do_load_custom_model('load_custom_model models/folder2/model2.npy'.split()) == 0

def test_error_save_model(app_instance):
    assert app_instance.do_save_model('save_model Z:|ASDPL'.split()) == -1

def test_make_custom_model_invalid_inputs(app_instance):
    assert app_instance.do_make_custom_model('make_custom_model'.split()) == -1
    assert app_instance.do_make_custom_model('make_custom_model -100 256 128'.split()) == -1
    assert app_instance.do_make_custom_model('make_custom_model 400 -256 128'.split()) == -1
    assert app_instance.do_make_custom_model('make_custom_model 400 256 not_a_number'.split()) == -1

def test_load_custom_model_invalid_inputs(app_instance):
    assert app_instance.do_load_custom_model('load_custom_model'.split()) == -1
    assert app_instance.do_load_custom_model('load_custom_model nonexistent_file.npy'.split()) == -1
    assert app_instance.do_load_custom_model('load_custom_model /invalid_path/file.npy'.split()) == -1

def test_valid_save_model(app_instance):
    assert app_instance.do_save_model('save_model models/folder2/saved_model.npy'.split()) == 0

def test_predict_edge_cases(app_instance):
    assert app_instance.do_predict('predict'.split()) == -1
    assert app_instance.do_predict('predict invalid_path.png'.split()) == -1
    assert app_instance.do_predict('predict /empty_folder/image.png'.split()) == -1 

def test_model_reinitialization(app_instance):
    previous_nn = copy.deepcopy(app_instance.nn)

    app_instance.do_make_custom_model('make_custom_model 400 200 100'.split())
    assert app_instance.nn != previous_nn 

    app_instance.nn = copy.deepcopy(previous_nn)  
    assert all(
        (w1 == w2).all()
        for w1, w2 in zip(app_instance.nn.weights_input_hidden, previous_nn.weights_input_hidden)
    ) and (app_instance.nn.weights_output_hidden == previous_nn.weights_output_hidden).all()


def test_edge_cases_training(app_instance):
    assert app_instance.do_train_on_random_images('train_on_random_images 10 -1 0.01'.split()) == -1
    assert app_instance.do_train_on_random_images('train_on_random_images 10 1 -0.01'.split()) == -1
    assert app_instance.do_train_on_random_images('train_on_random_images 10 1 0'.split()) == -1

def test_load_model_and_predict(app_instance):
    app_instance.do_load_default_model()
    app_instance.do_save_model('save_model models/test_model.npy'.split())
    assert app_instance.do_load_custom_model('load_custom_model models/test_model.npy'.split()) == 0

    test_cases = generate_standart_test_cases()
    cnt_right = count_of_right(test_cases, app_instance)
    assert cnt_right >= 8 

def test_model_reusability(app_instance):
    app_instance.do_make_custom_model('make_custom_model 128 64 32'.split())
    assert app_instance.do_train_on_random_images('train_on_random_images 1000 1 0.01'.split()) == 0

    first_model_predictions = count_of_right(generate_standart_test_cases(), app_instance)
    assert first_model_predictions > 0

    app_instance.do_make_custom_model('make_custom_model 256 128'.split())
    assert app_instance.do_train_on_random_images('train_on_random_images 1000 1 0.01'.split()) == 0

    second_model_predictions = count_of_right(generate_standart_test_cases(), app_instance)
    assert second_model_predictions > 0

    assert first_model_predictions != second_model_predictions

