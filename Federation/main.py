import os
import torch
from torchvision import models


def load_weights(file_path):
    return torch.load(file_path)


def federate_models(model, weights_list):
    # Get the state_dict of the model as a template
    federated_state_dict = model.state_dict()

    # Initialize a dictionary to accumulate the weights
    for key in federated_state_dict.keys():
        federated_state_dict[key] = sum([weights[key] for weights in weights_list]) / len(weights_list)

    return federated_state_dict


def main(weights_folder, output_file):
    # Initialize a ResNet50 model
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 9)  # Adjust to the number of classes

    # Get all .pth files from the weights folder
    weights_files = [os.path.join(weights_folder, file) for file in os.listdir(weights_folder) if file.endswith('.pth')]

    # Load all model weights
    weights_list = [load_weights(file) for file in weights_files]

    # Federate the weights
    federated_weights = federate_models(model, weights_list)

    # Load federated weights into the model
    model.load_state_dict(federated_weights)

    # Save the federated model
    torch.save(model.state_dict(), output_file)
    print(f"Federated model saved to {output_file}")


if __name__ == "__main__":
    weights_folder_path = "weights"  # Replace with your folder path
    output_file_path = "federated_model.pth"
    main(weights_folder_path, output_file_path)
