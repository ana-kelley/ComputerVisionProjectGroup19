import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

# Classes
classes = ('happy', 'sad', 'neutral', 'surprise', 'disgust')

# Root directory
root_directory = "path/to/EmotionDatasetImageFolder"

# Map to store images and their class labels
image_data = []

# Counters for accuracy calculation
total_iterations = 0
correct_guesses = 0

# Go through all folders
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)

    # Go through images of one folder
    if os.path.isdir(folder_path):
        image_files = os.listdir(folder_path)

        for i, image_file in enumerate(image_files[:100]):
            image_path = os.path.join(folder_path, image_file)

            # Read image and assign a class to it
            image = cv2.imread(image_path)
            random_class = random.choice(classes)

            # Check if random class matches actual class
            if random_class == folder_name:
                correct_guesses += 1

            # Check if the image was read successfully
            if image is not None:
                # Append the image and its label (folder_name) to the array
                image_data.append((folder_name, image, random_class))

                # Add iteration
                total_iterations += 1
            else:
                print(f"Failed to load {image_path}")

# Calculate accuracy
accuracy = correct_guesses / total_iterations * 100
print(f"Accuracy: {accuracy:.2f}%")

# Randomly select 12 images from image_data
selected_images = random.sample(image_data, 12)

# Set grid dimensions (3 rows, 4 columns)
rows, cols = 3, 4

# Create a figure to display images
fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

# Display each image in the grid
for i, ax in enumerate(axes.flat):
    folder_name, image, random_class = selected_images[i]

    # Convert BGR (OpenCV) to RGB (Matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show image and annotate with its folder_name and random class
    ax.imshow(image_rgb)
    ax.axis('off')  # Remove axes
    ax.set_title(f"True: {folder_name}\nPred: {random_class}", fontsize=10, loc='center')

    # Add grid lines
    ax.grid(True, color='white', linestyle='-', linewidth=0.5)

# Annotate the figure with the accuracy score
fig.text(0.5, 0.92, f"Accuracy: {accuracy:.2f}%", ha='center', fontsize=16, color='blue', fontweight='bold')

# Adjust layout and show the grid
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust spacing between images
plt.show()
