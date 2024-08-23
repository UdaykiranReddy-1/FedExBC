import tensorflow as tf
import pandas as pd
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam


PATH_TO_CSV_FILE = "../Dataset/ISIC_2019_Training_Input/split_part_2.csv"
PATH_TO_PRETRAINED_MODEL = "../PreTraining/resnet50_pretrained_model.h5"
PATH_TO_IMAGE_FOLDER = "../Dataset/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


# Load CSV data
data = pd.read_csv(PATH_TO_CSV_FILE)

# Load mapped class labels and process data
with open('models.json') as json_file:
    label_map = json.load(json_file)
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


# Load the pre-trained model
model = tf.keras.models.load_model(PATH_TO_PRETRAINED_MODEL)

# Unfreeze some layers in the base model
# Here we unfreeze the last few layers. Adjust the number of layers to unfreeze based on your requirements.
for layer in model.layers[-30:]:  # Unfreezing the last 30 layers for fine-tuning
    layer.trainable = True

# Re-compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the new dataset
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_val, y_val),
          epochs=10)

# Save the fine-tuned model
model.save('fine_tuned_model_1.h5')

