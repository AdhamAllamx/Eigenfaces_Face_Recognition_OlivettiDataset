import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Path to the folder containing facial images
dataset_folder = "./lfw"

# Initialize empty lists for data and targets
data = []
target = []

# Initialize label encoder
label_encoder = LabelEncoder()

# Iterate through the images folder
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        # Read each image file
        image_path = os.path.join(root, file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        
        # Resize the image to a fixed size
        img = cv2.resize(img, (64, 64))
        
        # Flatten the image into a 1D array and append to data
        data.append(img.flatten())
        
        # Extract the label from the folder name or file name
        label = os.path.basename(root)
        
        # Append the label to target
        target.append(label)

# Fit label encoder to the list of unique labels
label_encoder.fit(target)

# Transform labels to unique integer IDs
target = label_encoder.transform(target)

# Convert data and target lists to NumPy arrays
data = np.array(data)
target = np.array(target)

# Save data and target arrays into NumPy binary files
np.save("lfw_data.npy", data)
np.save("lfw_target.npy", target)

print("Dataset created successfully.")
