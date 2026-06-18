# NOTE regarding paths to relevant directories and files; for this script and ALL subsequent scripts, ensure you provide the appropriate paths (e.g., for images, labels,  train.txt file, .yaml file, etc.) 
## Within Anaconda Prompt, you can execute the below script by 1) setting your working directory, 2) initializing a conda environment, and then 3) entering "python 1_relabel_train.py" and pressing enter.
### Refer to the associated article and Supplemental Table 1 for descriptions of each script's intended usage and parameters. 

import os

train_txt_path = r"C:\Users\lchen\Desktop\master_combined_compiled\train.txt"
images_dir = r"C:\Users\lchen\Desktop\master_combined_compiled\images\train"
labels_dir = r"C:\Users\lchen\Desktop\master_combined_compiled\labels\train"

# Mapping of numeric labels to descriptive labels; change to reflect the behavioural categories ("classes") in your project 
label_mapping = {
    0: "T1_FEED",
    1: "T1_LOCO",
    2: "T1_OBMAN",
    3: "T1_REST",
    4: "T1_STEREO"
}

# Function to read annotation file and extract label information
def get_label_from_annotation(annotation_path):
    try:
        with open(annotation_path, 'r', encoding='utf-8') as file:
            label_str = file.readline().split()[0]
            label = int(float(label_str))  # Convert safely to int
            return label_mapping.get(label, "Unknown")
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Skipping {annotation_path} - {e}")
        return None  # Skip missing or corrupted annotation files

# Scan images directory for all PNG files
image_files = [f for f in os.listdir(images_dir) if f.endswith('.PNG')]

# Process images and update train.txt
updated_lines = []
for image_file in image_files:
    image_path = os.path.join(images_dir, image_file)
    annotation_file = image_file.replace('.PNG', '.txt')
    annotation_path = os.path.join(labels_dir, annotation_file)

    label_desc = get_label_from_annotation(annotation_path)
    if label_desc:  # Only include valid entries
        updated_lines.append(f"{image_path} {label_desc}\n")

# Write to train.txt
with open(train_txt_path, 'w', encoding='utf-8') as train_file:
    train_file.writelines(updated_lines)

print(f"✅ train.txt has been updated with {len(updated_lines)} valid entries.")
