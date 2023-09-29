# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model('model/cifar10_model.h5')

# # Load an image for prediction (you can replace this with your own image)
# # Make sure the image size matches the input shape of the model
# image = ...  # Load your image here

# # Preprocess the image (resize, normalize, etc.) to match model input
# # Replace this with your own preprocessing logic

# # Make a prediction
# prediction = model.predict(np.expand_dims(image, axis=0))

# # Get the predicted class label
# = np.argmax(prediction)

# # Print the result
# print(f"Predicted class: {predicted_class}")
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

def predict_image(cifar10_model, image_path):

    # Load the trained model
    model = load_model('model/cifar10_model.h5')

    # Load an image for prediction (you need to specify the image path)
    # Make sure the image size matches the input shape of the model
    # Load an image for prediction (you need to specify the image path)
    # Make sure the image size matches the input shape of the model
    image_path = "Image/gfx100s_sample_04_thum-1.jpg"

    # Load and preprocess the image
    img = keras_image.load_img(image_path, target_size=(32, 32))  # Adjust the target size as needed
    image = keras_image.img_to_array(img)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image data if your model expects it

    # Make a prediction
    prediction = model.predict(image)

    # Get the predicted class label
    predicted_class = np.argmax(prediction)

    # Print the result
    print(f"Predicted class: {predicted_class}")
    # Your prediction code here
    return predicted_class
