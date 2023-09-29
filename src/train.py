import tensorflow as tf
import sys
import os

# Add the project's root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preparation import load_cifar10_data
from src.model import create_cnn_model

def train_model(model, x_train, y_train,x_test, y_test, epochs=1, batch_size=64):

    # # Load CIFAR-10 data
    # x_train, y_train, x_test, y_test = load_cifar10_data()

    # Create and compile the model
    # model = create_cnn_model(x_train[0].shape)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    #model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.fit(x_train, y_train, epochs, batch_size, validation_data=(x_test, y_test))

    # Save the trained model
    model.save('model/cifar10_model.h5')
def train_model():

    # Load CIFAR-10 data
    x_train, y_train, x_test, y_test = load_cifar10_data()

    # Create and compile the model
    model = create_cnn_model(x_train[0].shape)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    #model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))

    # Save the trained model
    model.save('model/cifar10_model.h5')
# def train_model(model, x_train, y_train, x_test, y_test, epochs=1, batch_size=64):
#     # Your training code here
#     model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
#     return model  # Return the trained model

