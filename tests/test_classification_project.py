import pytest
import tensorflow as tf
import numpy as np
import sys
import os

# Add the project's root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

tf.keras.backend.clear_session()

from src.data_preparation import load_cifar10_data
from src.model import create_cnn_model
from src.train import train_model
from src.predict import predict_image

num_samples_for_testing = 1000 

@pytest.fixture
def cifar10_data():
    x_train, y_train, x_test, y_test = load_cifar10_data()
    x_test_subset, y_test_subset = x_test[:num_samples_for_testing], y_test[:num_samples_for_testing]
    return x_train, y_train, x_test_subset, y_test_subset

@pytest.fixture
def cifar10_model(cifar10_data):
    train_model()
    model = train_model()
    return model

def test_data_preparation(cifar10_data):
    x_train, y_train, x_test, y_test = cifar10_data
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert x_train.shape[1:] == (32, 32, 3)
    assert x_test.shape[1:] == (32, 32, 3)
    assert y_train.shape[1] == 10
    assert y_test.shape[1] == 10

def test_model_creation():
    model = create_cnn_model(input_shape=(32, 32, 3))
    assert isinstance(model, tf.keras.models.Sequential)
    assert model.layers

def test_model_training(cifar10_data, cifar10_model):
    x_train, y_train, x_test, y_test = cifar10_data
    if cifar10_model:
        _, accuracy = cifar10_model.evaluate(x_test, y_test)
        assert accuracy > 0.7  # Adjust as needed

def test_model_prediction(cifar10_model):
    test_image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    predicted_class = predict_image(cifar10_model, test_image)
    assert 0 <= predicted_class < 10

if __name__ == '__main__':
    # Redirect output to a file and display it in the console
    output_file = open('test_output.txt', 'w')
    pytest_args = ["-o", "tee", "test_output.txt"]
    pytest_args.extend(sys.argv[1:])  # Include any command-line arguments passed to the script
    pytest.main(pytest_args, plugins=[output_file])
    output_file.close()
