import os
from src.predict import predict_image
from src.train import train_model

# Get the directory where this script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the directory containing the images relative to the script's location
image_directory = os.path.join(script_directory, 'Image')

# Specify the output file path
output_file_path = 'output.txt'  # Change this to your desired output file path

# Initialize an empty list to store predicted classes and their corresponding file names
predicted_classes = []

# Function to process images
def process_images():
    # Clear the previous predicted classes
    predicted_classes.clear()

    # Iterate over the image files in the directory
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            # Construct the full path to the image file
            image_path = os.path.join(image_directory, filename)

            # Call the predict_image function with the image data
            predicted_class = predict_image(train_model, image_path)

            # Append the predicted class and filename to the list
            predicted_classes.append((filename, predicted_class))

# Run the process_images function initially
process_images()

# Write the predicted classes to the output file (if needed)
with open(output_file_path, 'w') as output_file:
    for filename, predicted_class in predicted_classes:
        output_file.write(f"File: {filename}, Predicted class: {predicted_class}\n")

# Print the results (if needed)
for filename, predicted_class in predicted_classes:
    print(f"File: {filename}, Predicted class: {predicted_class}")
