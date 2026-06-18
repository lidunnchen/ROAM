### Refer to the associated article and Supplemental Table 1 for descriptions of each script's intended usage and parameters. 

import os

# Define the paths to the images and labels folders for your project 
images_folder = r"C:\Users\lchen\Desktop\master_combined_compiled\images\train"
labels_folder = r"C:\Users\lchen\Desktop\master_combined_compiled\labels\train"

# Ensure the folders exist
if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
    print("Error: One or both folders do not exist. Please check the paths.")
    exit()

# Get the list of image files and label files
image_files = [f for f in os.listdir(images_folder) if f.endswith('.PNG')]
label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith('.txt')}

# Print debug information
print(f"Total images found: {len(image_files)}")
print(f"Total labels found: {len(label_files)}")

# Check for images without labels
images_without_labels = [f for f in image_files if os.path.splitext(f)[0] not in label_files]
print(f"Images without labels: {len(images_without_labels)}")

# Proceed to delete image files without a corresponding label file
for image_file in images_without_labels:
    image_path = os.path.join(images_folder, image_file)
    os.remove(image_path)  # Delete the file
    print(f"Deleted: {image_path}")

print("Cleanup complete.")
