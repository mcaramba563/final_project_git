import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import Perceptron

@pytest.fixture
def perceptron_instance():
    return Perceptron()

def test_gen_weights(perceptron_instance):
    perceptron_instance.gen_weights()
    assert len(perceptron_instance.weights_input_hidden) == len(perceptron_instance.hidden_layers_size)
    for i, weights in enumerate(perceptron_instance.weights_input_hidden):
        if i == 0:
            assert weights.shape == (perceptron_instance.input_size, perceptron_instance.hidden_layers_size[0])
        else:
            assert weights.shape == (perceptron_instance.hidden_layers_size[i - 1], perceptron_instance.hidden_layers_size[i])
    assert perceptron_instance.weights_output_hidden.shape == (perceptron_instance.hidden_layers_size[-1], perceptron_instance.output_size)

def test_bias_gen(perceptron_instance):
    perceptron_instance.bias_gen()
    assert len(perceptron_instance.bias_input_hidden) == len(perceptron_instance.hidden_layers_size)
    for i, bias in enumerate(perceptron_instance.bias_input_hidden):
        assert bias.shape == (1, perceptron_instance.hidden_layers_size[i])

def test_tanh(perceptron_instance):
    x = np.array([-1, 0, 1])
    expected = np.tanh(x)
    assert np.allclose(perceptron_instance.tanh(x), expected)

def test_tanh_derivative(perceptron_instance):
    x = np.array([-1, 0, 1])
    expected = 1 - np.tanh(x) ** 2
    assert np.allclose(perceptron_instance.tanh_derivative(x), expected)

def test_softmax(perceptron_instance):
    x = np.array([1, 2, 3])
    expected = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    assert np.allclose(perceptron_instance.softmax(x), expected)

def test_forward(perceptron_instance):
    X = np.random.rand(perceptron_instance.input_size)
    output = perceptron_instance.forward(X)
    assert output.shape == (1, perceptron_instance.output_size)
    assert np.isclose(np.sum(output), 1.0)

def test_backprop(perceptron_instance):
    X = np.random.rand(perceptron_instance.input_size)
    y = np.zeros(perceptron_instance.output_size)
    y[0] = 1 
    perceptron_instance.forward(X)
    perceptron_instance.backprop(y)

def test_train(perceptron_instance):
    X = np.random.rand(10, perceptron_instance.input_size) 
    y = np.random.randint(0, perceptron_instance.output_size, size=10)
    perceptron_instance.train(X, y, epochs=1, learning_rate=0.01)

def test_predict(perceptron_instance):
    X = np.random.rand(perceptron_instance.input_size)
    prediction = perceptron_instance.predict(X)
    assert 0 <= prediction < perceptron_instance.output_size

def test_save_and_load_model(tmp_path, perceptron_instance):
    model_path = tmp_path / "model.npy"
    perceptron_instance.save_model(str(model_path))
    assert model_path.exists()

    new_perceptron = Perceptron()
    new_perceptron.load_model(str(model_path))

    for w1, w2 in zip(perceptron_instance.weights_input_hidden, new_perceptron.weights_input_hidden):
        assert np.allclose(w1, w2)
    assert np.allclose(perceptron_instance.weights_output_hidden, new_perceptron.weights_output_hidden)
    for b1, b2 in zip(perceptron_instance.bias_input_hidden, new_perceptron.bias_input_hidden):
        assert np.allclose(b1, b2)