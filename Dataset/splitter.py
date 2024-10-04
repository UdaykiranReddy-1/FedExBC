import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

global remaining_data

# Load metadata
metadata = pd.read_csv('ISIC_2019_Training_Input/ISIC_2019_Training_GroundTruth.csv')

# Create a column for the primary label
# Assuming one-hot encoding, you can get the column with the max value
metadata['primary_label'] = metadata[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']].idxmax(axis=1)

# Create a list to store the data splits
splits = []

# First split: Separate out 10,000 images
splitter = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=42)

for train_index, split_index in splitter.split(metadata, metadata['primary_label']):
    train_set = metadata.iloc[train_index]
    large_split = metadata.iloc[split_index]
    # Append the large split to the splits list
    splits.append(large_split)
    # Remaining data to be split into four parts
    remaining_data = train_set

# Calculate the size of each remaining split (should be about 3,750 images per part)
remaining_size = len(remaining_data)
split_size = remaining_size // 4

# Split the remaining data into four parts
splitter = StratifiedShuffleSplit(n_splits=4, test_size=split_size, random_state=42)

for _, split_index in splitter.split(remaining_data, remaining_data['primary_label']):
    split_part = remaining_data.iloc[split_index]
    splits.append(split_part)

# Save each split to a separate CSV file
for i, split in enumerate(splits):
    split = split.drop('primary_label', axis=1)
    split.to_csv(f'split_part_{i + 1}.csv', index=False)
