# There are many methods to separate datasets into training and test splits (e.g., nested k-fold cross validation). We provide a simple method to introductory users that conducts an 80:20 split. However, it is important also to test model performance on a holdout dataset that has not been seen during the training stage. 
## Refer to the associated article and Supplemental Table 1 for descriptions of each script's intended usage and parameters. 

import os
import shutil
import random

# Define source and destination folders
images_src_folder = r"C:\Users\lchen\Desktop\master_combined_compiled\images\train"
labels_src_folder = r"C:\Users\lchen\Desktop\master_combined_compiled\labels\train"
train_images_folder = r"C:\Users\lchen\Desktop\master_combined_compiled\train\images"
val_images_folder = r"C:\Users\lchen\Desktop\master_combined_compiled\val\images"
train_labels_folder = r"C:\Users\lchen\Desktop\master_combined_compiled\train\labels"
val_labels_folder = r"C:\Users\lchen\Desktop\master_combined_compiled\val\labels"

# Ensure destination folders exist
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Get sorted lists of image and label files
image_files = sorted([f for f in os.listdir(images_src_folder) if f.lower().endswith('.png')])
label_files = sorted([f for f in os.listdir(labels_src_folder) if f.lower().endswith('.txt') and f != 'classes.txt'])

# Check if the number of images matches the number of labels
if len(image_files) != len(label_files):
    print("Warning: Number of images and labels do not match!")
    print(f"Images: {len(image_files)}, Labels: {len(label_files)}")

# Shuffle data for randomness
combined = list(zip(image_files, label_files))
random.shuffle(combined)
image_files, label_files = zip(*combined)

# Split data into train (80%) and val (20%)
val_count = int(0.2 * len(image_files))
val_image_files = image_files[:val_count]
val_label_files = label_files[:val_count]
train_image_files = image_files[val_count:]
train_label_files = label_files[val_count:]

# Copy val data
for image_file, label_file in zip(val_image_files, val_label_files):
    shutil.copy(os.path.join(images_src_folder, image_file), os.path.join(val_images_folder, image_file))
    shutil.copy(os.path.join(labels_src_folder, label_file), os.path.join(val_labels_folder, label_file))
    print(f"Copied to val: {image_file}, {label_file}")

# Copy train data
for image_file, label_file in zip(train_image_files, train_label_files):
    shutil.copy(os.path.join(images_src_folder, image_file), os.path.join(train_images_folder, image_file))
    shutil.copy(os.path.join(labels_src_folder, label_file), os.path.join(train_labels_folder, label_file))
    print(f"Copied to train: {image_file}, {label_file}")

print("Data split complete.")
