# This script will output the total number of labels for each class of interest, providing the user with information regarding the data distribution for each class on which predictions will be made. 
## Adjust the label_counts category by adding or removing the # of behavioural classes in your project. There were five behaviours of interest in our study; thus, we start with a count of zero for classes 0, 1, 2, 3, and 4. 
### Refer to the associated article and Supplemental Table 1 for descriptions of each script's intended usage and parameters. 


import os

# Path to the labels directory
labels_folder = r"C:\Users\lchen\Desktop\master_combined_compiled\labels\train"

# Initialize counts for each label class
label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

# Process each label file
for filename in os.listdir(labels_folder):
    if filename.endswith('.txt'):
        label_path = os.path.join(labels_folder, filename)

        # Read the label file
        with open(label_path, 'r') as file:
            lines = file.readlines()

        # Count occurrences of each label in the file
        for line in lines:
            parts = line.split()
            if len(parts) > 0:
                # Convert to float first, then to int (to handle cases like '2.0')
                label = int(float(parts[0]))  
                if label in label_counts:
                    label_counts[label] += 1

# Print the number of instances for each class
for label, count in label_counts.items():
    print(f"Class {label}: {count} samples")
