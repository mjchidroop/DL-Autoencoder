# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
The goal is to build a Convolutional Autoencoder that can effectively remove Gaussian noise from grayscale images. The model learns to reconstruct clean images from noisy inputs by compressing the image into a latent representation and then decoding it back to the original clean version.

**Dataset**: MNIST dataset containing 70,000 grayscale handwritten digit images (0-9) of size 28×28 pixels.
- Training set: 60,000 images
- Test set: 10,000 images
- Pixel range: [0, 1] after normalization

## DESIGN STEPS

### STEP 1: 
Load MNIST dataset using torchvision and create DataLoaders with batch size 128.

### STEP 2: 
Define add_noise() function to add Gaussian noise with noise factor 0.5 to images.

### STEP 3: 
Build encoder with 3 Conv2d layers (1→16→32→64) with ReLU activation.

### STEP 4: 
Build decoder with 3 ConvTranspose2d layers (64→32→16→1) with Sigmoid final activation.

### STEP 5: 
Train model using MSELoss, Adam optimizer (lr=0.001), and noise factor 0.5 for 10 epochs.

### STEP 6: 
Evaluate and visualize denoising results on test data, showing original vs noisy vs denoised images.

## PROGRAM

### Name: Chidroop M J

### Register Number: 212225240029

**Import Libraries**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

**Load Dataset**
```python
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

**Add Noise Function**
```python
def add_noise(img, noise_factor=0.5):
    noisy_img = img + noise_factor * torch.randn_like(img)
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img

sample_img, _ = train_dataset[0]
noisy_sample = add_noise(sample_img, 0.5)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_sample.squeeze(), cmap='gray')
plt.title('Noisy Image (factor=0.5)')
plt.axis('off')
plt.show()
```

**Autoencoder Definition**
```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if decoded.shape[-1] != x.shape[-1]:
            decoded = F.interpolate(decoded, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return decoded

model = DenoisingAutoencoder().to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

**Initialize Model and Training Setup**
```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
noise_factor = 0.5

train_losses = []
```

**Training Function**
```python
def train_model(model, train_loader, criterion, optimizer, num_epochs, noise_factor):
    model.train()
    train_losses = []
    
    print("Training Started...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            noisy_data = add_noise(data, noise_factor)
            
            outputs = model(noisy_data)
            loss = criterion(outputs, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')
    
    print("Training Completed!")
    return train_losses

train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs, noise_factor)
```

**Visualization Function**
```python
def visualize_results(model, test_loader, num_images=8, noise_factor=0.5):
    model.eval()
    
    data, _ = next(iter(test_loader))
    data = data[:num_images].to(device)
    
    noisy_data = add_noise(data, noise_factor)
    
    with torch.no_grad():
        denoised_data = model(noisy_data)
    
    data = data.cpu().numpy()
    noisy_data = noisy_data.cpu().numpy()
    denoised_data = denoised_data.cpu().numpy()
    
    fig, axes = plt.subplots(3, num_images, figsize=(15, 6))
    
    for i in range(num_images):
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(noisy_data[i].squeeze(), cmap='gray')
        axes[1, i].set_title('Noisy')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(denoised_data[i].squeeze(), cmap='gray')
        axes[2, i].set_title('Denoised')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Autoencoder Denoising Results', fontsize=14)
    plt.subplots_adjust(top=0.88)
    plt.show()
```

**Model Summary**
```python
print("="*50)
print("MODEL SUMMARY")
print("="*50)
print(f"Architecture: Convolutional Autoencoder for Image Denoising")
print(f"Input shape: (batch_size, 1, 28, 28)")
print(f"Output shape: (batch_size, 1, 28, 28)")
print(f"Total parameters: {total_params:,}")
print("="*50)

print("\nENCODER LAYERS:")
print("-"*50)
print(f"Layer 1: Conv2d(1, 16, kernel_size=3, stride=2, padding=1) → Output: (16, 14, 14)")
print(f"Activation: ReLU")
print(f"Layer 2: Conv2d(16, 32, kernel_size=3, stride=2, padding=1) → Output: (32, 7, 7)")
print(f"Activation: ReLU")
print(f"Layer 3: Conv2d(32, 64, kernel_size=3, stride=2, padding=1) → Output: (64, 4, 4)")
print(f"Activation: ReLU")

print("\nDECODER LAYERS:")
print("-"*50)
print(f"Layer 1: ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) → Output: (32, 8, 8)")
print(f"Activation: ReLU")
print(f"Layer 2: ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1) → Output: (16, 16, 16)")
print(f"Activation: ReLU")
print(f"Layer 3: ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1) → Output: (1, 32, 32)")
print(f"Activation: Sigmoid")
print(f"Resize: Interpolate to (28, 28) using bilinear interpolation")

print("\nLOSS FUNCTION: MSELoss")
print(f"OPTIMIZER: Adam (lr=0.001)")
print(f"NOISE FACTOR: 0.5")
print(f"EPOCHS: {num_epochs}")
```

**Plot Training Loss**
```python
plt.figure(figsize=(10, 5))
plt.plot(train_losses, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.show()
```

**Visualize Results**
```python
visualize_results(model, test_loader)
```

**Evaluate Model**
```python
def evaluate_model(model, test_loader, noise_factor=0.5):
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            noisy_data = add_noise(data, noise_factor)
            denoised_data = model(noisy_data)
            
            mse = F.mse_loss(denoised_data, data)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            total_mse += mse.item() * data.size(0)
            total_psnr += psnr.item() * data.size(0)
            num_samples += data.size(0)
    
    avg_mse = total_mse / num_samples
    avg_psnr = total_psnr / num_samples
    
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    return avg_mse, avg_psnr

avg_mse, avg_psnr = evaluate_model(model, test_loader)
```

### OUTPUT

### Model Summary
```
==================================================
MODEL SUMMARY
==================================================
Architecture: Convolutional Autoencoder for Image Denoising
Input shape: (batch_size, 1, 28, 28)
Output shape: (batch_size, 1, 28, 28)
Total parameters: 46,529
==================================================

ENCODER LAYERS:
--------------------------------------------------
Layer 1: Conv2d(1, 16, kernel_size=3, stride=2, padding=1) → Output: (16, 14, 14)
Activation: ReLU
Layer 2: Conv2d(16, 32, kernel_size=3, stride=2, padding=1) → Output: (32, 7, 7)
Activation: ReLU
Layer 3: Conv2d(32, 64, kernel_size=3, stride=2, padding=1) → Output: (64, 4, 4)
Activation: ReLU

DECODER LAYERS:
--------------------------------------------------
Layer 1: ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) → Output: (32, 8, 8)
Activation: ReLU
Layer 2: ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1) → Output: (16, 16, 16)
Activation: ReLU
Layer 3: ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1) → Output: (1, 32, 32)
Activation: Sigmoid
Resize: Interpolate to (28, 28) using bilinear interpolation

LOSS FUNCTION: MSELoss
OPTIMIZER: Adam (lr=0.001)
NOISE FACTOR: 0.5
EPOCHS: 10

```

### Training loss
<img width="855" height="470" alt="image" src="https://github.com/user-attachments/assets/49bcda58-e627-415b-bd59-45ebef76dd95" />

## Original vs Noisy vs Reconstructed Image
<img width="1474" height="593" alt="image" src="https://github.com/user-attachments/assets/0cdfa02d-63b9-45b3-8921-d1429ea375fd" />

## RESULT
The Convolutional Autoencoder was successfully implemented for image denoising on the MNIST dataset. The model:
- Successfully learned to remove Gaussian noise (factor=0.5) from images
- Achieved good reconstruction quality with visible denoising results
- Final training loss: 0.0129
- Test MSE: 0.012535
- Test PSNR: 19.03 dB

The architecture effectively compressed images through the encoder and reconstructed clean versions through the decoder, demonstrating the capability of autoencoders for noise removal in image processing applications.
