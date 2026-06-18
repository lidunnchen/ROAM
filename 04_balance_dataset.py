# It is often times important to balance your dataset so that the model does not become biased during training. The number of samples (images and associated labels) should be approximtely the same across behavioural classes of interest. 
## Adust the "class_counts" argument below to reflect the total number of samples in each group. This information is important to provide in the script so that we can properly adjust/balance the dataset.
### If you are not sure of how many sample labels there are for each behaviour, you can apply the included script, "samples_distribution_check.py" to determine that information. 
### Refer to the associated article and Supplemental Table 1 for descriptions of each script's intended usage and parameters. 


import os
import shutil
import random

# Directories for images and labels
image_dir = "C:/Users/lchen/Desktop/master_combined_compiled/images/train"
label_dir = "C:/Users/lchen/Desktop/master_combined_compiled/labels/train"
val_extra_images_folder = "C:/Users/lchen/Desktop/master_combined_compiled/val_extra/images"
val_extra_labels_folder = "C:/Users/lchen/Desktop/master_combined_compiled/val_extra/labels"

# Ensure the 'val_extra' folders exist
os.makedirs(val_extra_images_folder, exist_ok=True)
os.makedirs(val_extra_labels_folder, exist_ok=True)

# Read the train.txt file to get the image list and corresponding labels
train_txt_file = "C:/Users/lchen/Desktop/master_combined_compiled/train.txt"
with open(train_txt_file, 'r') as file:
    lines = file.readlines()

# Class counts (adjust to match your dataset's class structure)
class_counts = {
    "T1_FEED": 18996,
    "T1_LOCO": 11795,
    "T1_OBMAN": 17824,
    "T1_REST": 13714,
    "T1_STEREO": 10529
}

# Find the class with the minimum count (T1_STEREO in this case)
min_class_count = min(class_counts.values())

# Create a dictionary to store images by class
images_by_class = {cls: [] for cls in class_counts}

# Sort images into their respective class lists
for line in lines:
    image_path, label_class = line.strip().split(' ')
    images_by_class[label_class].append(image_path)

# Now, sample the overrepresented classes to match the minimum class count
for label_class, image_paths in images_by_class.items():
    if len(image_paths) > min_class_count:
        images_by_class[label_class] = random.sample(image_paths, min_class_count)

# Move selected images and labels to val_extra
for label_class, image_paths in images_by_class.items():
    for image_path in image_paths:
        # Move image file
        image_filename = os.path.basename(image_path)
        shutil.move(image_path, os.path.join(val_extra_images_folder, image_filename))

        # Construct the corresponding label path (assumes label file is in 'labels/train')
        label_filename = image_filename.replace(".PNG", ".txt")  # Adjust the extension as needed
        label_path = os.path.join(label_dir, label_filename)

        # Move the label file if it exists
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(val_extra_labels_folder, label_filename))
        else:
            print(f"Warning: Label file for {image_filename} not found. Skipping.")
