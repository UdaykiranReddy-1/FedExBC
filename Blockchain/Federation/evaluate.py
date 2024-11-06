import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os


WEIGHTS_PATH = "federated_model.pth"


# Define the custom dataset
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


# Define the transformation (same as before)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the new dataset
img_dir = "../../Dataset/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
csv_file = "../../Dataset/ISIC_2019_Training_Input/split_part_5.csv"

dataset = SkinLesionDataset(csv_file_path=csv_file, img_dir_path=img_dir, transform_comp=transform)
dataset_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load the model
model = models.resnet50()  # ChatGPT had given pretrained param as false
model.fc = torch.nn.Linear(model.fc.in_features, 9)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in dataset_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        predictions = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold

        correct += (predictions == labels).sum().item()
        total += labels.numel()

accuracy = correct / total
print(f'Federated model accuracy: {accuracy:.4f}')

