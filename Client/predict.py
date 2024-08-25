import torch
from torchvision import transforms, models
from PIL import Image


WEIGHTS_PATH = "resnet50_skin_lesion.pth"


# Load the image
img_path = "path/to/image.jpg"
image = Image.open(img_path).convert("RGB")

# Apply the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = transform(image).unsqueeze(0)  # Add batch dimension

# Load the model and pass the image through it
model = models.resnet50()  # ChatGPT had given pretrained param as false
model.fc = torch.nn.Linear(model.fc.in_features, 9)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()

with torch.no_grad():
    outputs = model(image)  # Raw logits

# Apply sigmoid to get probabilities
probabilities = torch.sigmoid(outputs)

# Example output
print(probabilities)

predicted_classes = (probabilities > 0.5).int()
print(predicted_classes)
