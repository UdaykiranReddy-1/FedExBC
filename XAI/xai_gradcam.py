import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms

def generate_gradcam(model, input_image, target_layer):
    # Hook to save gradients of the target layer
    gradients = []
    def save_gradient(grad):
        gradients.append(grad)
    
    # Forward hook to capture target layer output
    target_output = None
    def forward_hook(module, input, output):
        nonlocal target_output
        target_output = output
        output.register_hook(save_gradient)
    
    # Register hooks on the target layer
    target_layer.register_forward_hook(forward_hook)

    # Forward pass
    model.eval()
    output = model(input_image)

    # Backward pass with a one-hot vector for the target class
    model.zero_grad()
    target_class = output.argmax().item()
    one_hot = torch.zeros(output.shape).to(output.device)
    one_hot[0][target_class] = 1
    output.backward(gradient=one_hot)
    
    # Generate Grad-CAM heatmap
    grad = gradients[0].detach().cpu().numpy()[0]
    weights = grad.mean(axis=(1, 2))
    cam = np.sum(weights * target_output[0].detach().cpu().numpy(), axis=0)
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    # Overlay on the input image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(input_image.cpu().numpy().transpose(1, 2, 0))
    overlay = overlay / np.max(overlay)
    
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()
    return overlay

# Example usage after loading final model and preparing an input image
# Assuming 'final_model' is your loaded aggregated model, and 'input_img' is a preprocessed input image tensor
target_layer = final_model.layer4[1].conv2  # Modify according to your model architecture
generate_gradcam(final_model, input_img.unsqueeze(0), target_layer)
