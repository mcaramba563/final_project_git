import pytest
from unittest.mock import Mock
import builtins
from app import *
import numpy as np

def count_of_right(test_cases):
    mock_print = Mock()
    builtins.print = mock_print
    cnt_right = 0
    for command, expected in test_cases:
        do_predict(command)
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

def test_valid_predict():
    test_cases = generate_standart_test_cases()

    cnt_right = count_of_right(test_cases)

    assert cnt_right == 10

def test_incorrect_predict():
    assert do_predict('predict images/mnist_png/test/9asdad/7.png'.split()) == -1
    assert do_predict('predict'.split()) == -1

def test_incorrect_train():
    assert do_train('train file_with_names.txt'.split()) == -1
    assert do_train('train file_with_names.txt 2'.split()) == -1
    assert do_train('train file_with_names.txt -1 0.01'.split()) == -1
    assert do_train('train asd.txt 1 0.01'.split()) == -1
    assert do_train('train file_with_incorrects_names.txt 1 0.01'.split()) == -1

def test_correct_train():
    assert do_train('train file_with_names.txt 1 0.01'.split()) == 0

def test_train_on_random_images():
    assert do_train_on_random_images('train_on_random_images 100'.split()) == -1
    assert do_train_on_random_images('train_on_random_images 100 1'.split()) == -1
    assert do_train_on_random_images('train_on_random_images 100 0 0.01'.split()) == -1
    assert do_train_on_random_images('train_on_random_images 0 10 0.01'.split()) == -1

    assert do_train_on_random_images('train_on_random_images 10 10 0.01'.split()) == 0

def test_make_custom_model():
    assert do_make_custom_model('make_custom_model 400 256 128'.split()) == 0
    assert do_train_on_random_images('train_on_random_images 20000 2 0.01'.split()) == 0

    test_cases = generate_standart_test_cases()
    assert count_of_right(test_cases) >= 7


def load_custom_model():
    assert do_load_custom_model('load_custom_model models/folder2/asd.npy'.split()) == 0

def test_error_save_model():
    assert do_save_model('save_model Z:|ASDPL'.split()) == -1



def test_make_custom_model_invalid_inputs():
    assert do_make_custom_model('make_custom_model'.split()) == -1
    assert do_make_custom_model('make_custom_model -100 256 128'.split()) == -1
    assert do_make_custom_model('make_custom_model 400 -256 128'.split()) == -1
    assert do_make_custom_model('make_custom_model 400 256 not_a_number'.split()) == -1

def test_load_custom_model_invalid_inputs():
    assert do_load_custom_model('load_custom_model'.split()) == -1
    assert do_load_custom_model('load_custom_model nonexistent_file.npy'.split()) == -1
    assert do_load_custom_model('load_custom_model /invalid_path/file.npy'.split()) == -1

def test_valid_save_model():
    assert do_save_model('save_model models/folder2/saved_model.npy'.split()) == 0

def test_predict_edge_cases():
    assert do_predict('predict'.split()) == -1
    assert do_predict('predict invalid_path.png'.split()) == -1
    assert do_predict('predict /empty_folder/image.png'.split()) == -1 

def test_train_on_empty_dataset():
    global get_random_pathes
    original_get_random_pathes = get_random_pathes
    def mock_get_random_pathes(n):
        return ([], [])
    get_random_pathes = mock_get_random_pathes

    assert do_train_on_random_images('train_on_random_images 10 1 0.01'.split()) == 0

    get_random_pathes = original_get_random_pathes

def test_model_reinitialization():
    global nn, default_model
    previous_nn = copy.deepcopy(nn)

    do_make_custom_model('make_custom_model 400 200 100'.split())
    assert nn != previous_nn 

    nn = copy.deepcopy(previous_nn)  
    assert all(
        (w1 == w2).all()
        for w1, w2 in zip(nn.weights_input_hidden, previous_nn.weights_input_hidden)
    ) and (nn.weights_output_hidden == previous_nn.weights_output_hidden).all()


def test_edge_cases_training():
    assert do_train_on_random_images('train_on_random_images 10 -1 0.01'.split()) == -1
    assert do_train_on_random_images('train_on_random_images 10 1 -0.01'.split()) == -1
    assert do_train_on_random_images('train_on_random_images 10 1 0'.split()) == -1

def test_load_model_and_predict():
    do_load_default_model()
    do_save_model('save_model models/test_model.npy'.split())
    assert do_load_custom_model('load_custom_model models/test_model.npy'.split()) == 0

    test_cases = generate_standart_test_cases()
    cnt_right = count_of_right(test_cases)
    assert cnt_right >= 8 

def test_model_reusability():
    do_make_custom_model('make_custom_model 128 64 32'.split())
    assert do_train_on_random_images('train_on_random_images 1000 1 0.01'.split()) == 0

    first_model_predictions = count_of_right(generate_standart_test_cases())
    assert first_model_predictions > 0

    do_make_custom_model('make_custom_model 256 128'.split())
    assert do_train_on_random_images('train_on_random_images 1000 1 0.01'.split()) == 0

    second_model_predictions = count_of_right(generate_standart_test_cases())
    assert second_model_predictions > 0

    assert first_model_predictions != second_model_predictions

