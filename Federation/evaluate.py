import tensorflow as tf
import pandas as pd
import numpy as np
import json


PATH_TO_CSV_FILE = "../Dataset/ISIC_2019_Training_Input/split_part_5.csv"
PATH_TO_IMAGE_FOLDER = "../Dataset/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


def load_images(data_row):
    images = []
    img_labels = []
    img_count = 0
    print()
    for _, row in data_row.iterrows():
        img_path = f'{PATH_TO_IMAGE_FOLDER}/{row["image"]}.jpg'
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img)
        img_labels.append(row['label'])
        img_count += 1
        print(LINE_UP, end=LINE_CLEAR)
        print(f"Pre-processing images: {img_count}")
    print("\nImage preprocessing completed")
    return np.array(images), np.array(img_labels)


def get_data_from_csv():
    # Load CSV data
    data = pd.read_csv(PATH_TO_CSV_FILE)

    # Load mapped class labels and process data
    with open('models.json') as json_file:
        label_map = json.load(json_file)
        data['label'] = data['primary_label'].map(label_map)

    return load_images(data)
