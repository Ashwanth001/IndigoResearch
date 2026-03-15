import pandas as pd
import os

# Define directories
output_dir = "data"
labels_file = os.path.join(output_dir, "labels_h5.csv")

train_output = os.path.join(output_dir, "train_labels.csv")
val_output = os.path.join(output_dir, "val_labels.csv")
test_output = os.path.join(output_dir, "test_labels.csv")

print(f"Starting Step 8: Splitting Into Train/Validation/Test Windows...")

if not os.path.exists(labels_file):
    print(f"Error: {labels_file} not found. Did you run Step 4?")
    exit(1)

# Load labels
labels = pd.read_csv(labels_file)

# Split according to plan:
# Training: up to 2012
# Validation: 2013
# Test: 2015
train_labels = labels[labels['year'].between(2000, 2012)]
val_labels = labels[labels['year'] == 2013]
test_labels = labels[labels['year'] == 2015]

# Save split labels
print(f"Saving splits to {train_output}, {val_output}, and {test_output}...")
train_labels.to_csv(train_output, index=False)
val_labels.to_csv(val_output, index=False)
test_labels.to_csv(test_output, index=False)

# Validation Statistics
print("\nValidation Statistics for Step 8:")
print(f"Train labels (2000-2012): {len(train_labels)} rows, balanced.")
print(f"Val labels (2013): {len(val_labels)} rows.")
print(f"Test labels (2015): {len(test_labels)} rows.")
print(f"Total entries: {len(train_labels) + len(val_labels) + len(test_labels)}")
print(f"Positive samples (Train): {(train_labels['label'] == 1).sum()}")
print(f"Negative samples (Train): {(train_labels['label'] == 0).sum()}")
