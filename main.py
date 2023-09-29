# from src.predict import predict_image


# # Call the predict_image function with your image data
# # Replace 'image_path' with the path to your image file
# image_path = 'classification_project\classification_project\Image\gfx100s_sample_04_thum-1.jpg'
# predicted_class = predict_image(image_path)
# print(f"Predicted class: {predicted_class}")
from src.predict import predict_image

# Call the predict_image function with your image data
# Replace 'image_path' with the path to your image file
image_path = '3345-IAC1\gfx100s_sample_04_thum-1.jpg'
predicted_class = predict_image(image_path)

# Specify the output file path
output_file_path = 'output.txt'  # Change this to your desired output file path

# Write the predicted class to the output file
with open(output_file_path, 'w') as output_file:
    output_file.write(f"Predicted class: {predicted_class}")

print(f"Predicted class: {predicted_class}")
