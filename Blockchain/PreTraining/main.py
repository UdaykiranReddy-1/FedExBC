import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import ResNet50_Weights
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os


class SkinLesionDataset(Dataset):
    def __init__(self, csv_file_path, img_dir_path, transform_comp=None):
        self.data = pd.read_csv(csv_file_path)
        self.img_dir = img_dir_path
        self.transform = transform_comp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name).convert("RGB")
        img_labels = self.data.iloc[idx, 1:].values.astype('float32')
        img_labels = torch.tensor(img_labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, img_labels


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_dir = "../../Dataset/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
csv_file = "../../Dataset/ISIC_2019_Training_Input/split_part_1.csv"

dataset = SkinLesionDataset(csv_file_path=csv_file, img_dir_path=img_dir, transform_comp=transform)

# Set a random seed for reproducibility (optional)
torch.manual_seed(42)

# Split the dataset into training and testing sets
train_size = int(0.9 * len(dataset))  # 90% for training
test_size = len(dataset) - train_size  # 10% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Initializing model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 9)  # 9 classes in your dataset

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f'Starting Epoch {epoch+1}')
    model.train()
    running_loss = 0.0

    print(f'Epoch {epoch+1}: Iterating through data')
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Evaluation phase
    print(f'Epoch {epoch+1}: Starting evaluation')
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predictions = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold

            correct += (predictions == labels).sum().item()
            total += labels.numel()

    accuracy = correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.4f}', end='\n\n')


torch.save(model.state_dict(), "pre_trained_model.pth")
