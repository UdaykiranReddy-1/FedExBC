import tensorflow as tf
import pandas as pd
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


PATH_TO_CSV_FILE = "../Dataset/ISIC_2019_Training_Input/split_part_1.csv"
PATH_TO_IMAGE_FOLDER = "../Dataset/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


# Load CSV data
data = pd.read_csv(PATH_TO_CSV_FILE)

# Map class labels to numbers and save them
labels = data['primary_label'].unique()
label_map = {label: i for i, label in enumerate(labels)}
with open('labels.json', 'w') as json_file:
    json.dump(label_map, json_file)
data['label'] = data['primary_label'].map(label_map)

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.1, stratify=data['label'])

# Image data generator for preprocessing
datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)


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


X_train, y_train = load_images(train_data)
X_val, y_val = load_images(val_data)


# Load the pre-trained ResNet50 model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(labels), activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=10)

# Save the model
model.save('resnet50_pretrained_model.h5')

