import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from evaluate import get_data_from_csv
 
WEIGHTS_FOLDER = 'Weights'
# Paths to fine-tuned models
model_paths = [
    'fine_tuned_model_1.h5',
    'fine_tuned_model_2.h5',
    'fine_tuned_model_3.h5'
    # Add more paths as needed
]

# Load all fine-tuned models
models = [tf.keras.models.load_model(f"{WEIGHTS_FOLDER}/{model_path}") for model_path in model_paths]

# Initialize a list to store the weights from each model
model_weights = [model.get_weights() for model in models]

# Average the weights across all models
average_weights = []

# Loop through each layer's weights
for weights_tuple in zip(*model_weights):
    # Average the weights for this layer across all models
    layer_average = np.mean(weights_tuple, axis=0)
    average_weights.append(layer_average)

# Load one model to use as the base for the federated model
federated_model = tf.keras.models.load_model(model_paths[0])

# Set the federated (averaged) weights to the model
federated_model.set_weights(average_weights)

# Compile the federated model
federated_model.compile(optimizer=Adam(learning_rate=1e-5),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])


# Evaluate the federated model
X_val, y_val = get_data_from_csv()
loss, accuracy = federated_model.evaluate(X_val, y_val)
print(f"Federated Model Accuracy: {accuracy * 100:.2f}%")
