import os
import time
from src.predict import predict_image
from src.train import train_model

# Get the directory where this script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the directory containing the images relative to the script's location
image_directory = os.path.join(script_directory, 'Image')

# Specify the output file path
output_file_path = 'output.txt'  # Change this to your desired output file path

# Initialize a set to keep track of processed file names
processed_files = set()

while True:
    # Get the list of image files in the directory with .jpg, .png, and .jpeg extensions
    image_files = [filename for filename in os.listdir(image_directory) if filename.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Filter out files that have already been processed
    new_files = [filename for filename in image_files if filename not in processed_files]
    
    if new_files:
        # Initialize an empty list to store predicted classes and their corresponding file names
        predicted_classes = []

        # Iterate over the new image files
        for filename in new_files:
            # Construct the full path to the image file
            image_path = os.path.join(image_directory, filename)

            # Call the predict_image function with the image data
            predicted_class = predict_image(train_model, image_path)

            # Append the predicted class and filename to the list
            predicted_classes.append((filename, predicted_class))

            # Add the filename to the set of processed files
            processed_files.add(filename)

        # Write the predicted classes to the output file
        with open(output_file_path, 'a') as output_file:
            for filename, predicted_class in predicted_classes:
                output_file.write(f"File: {filename}, Predicted class: {predicted_class}\n")

        # Print the results
        for filename, predicted_class in predicted_classes:
            print(f"File: {filename}, Predicted class: {predicted_class}")

    # Sleep for a while before checking again (adjust the sleep duration as needed)
    time.sleep(10)  # Sleep for 10 seconds before checking again
