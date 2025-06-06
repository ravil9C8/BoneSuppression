#%% load libraries; clear current gpu session
import warnings 
import os
warnings.filterwarnings('ignore',category=FutureWarning) #because of numpy version

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure,MultiScaleStructuralSimilarityIndexMeasure
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

img_rows = 1024
img_cols = 1024
img_channels = 1

# Paths
source_path = "/home/ravil/.cache/kagglehub/datasets/hmchuong/xray-bone-shadow-supression/versions/1/augmented/augmented/source"
target_path = "/home/ravil/.cache/kagglehub/datasets/hmchuong/xray-bone-shadow-supression/versions/1/augmented/augmented/target"

def apply_Clahe(img):
    img_np = np.array(img)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    img_np = clahe.apply(img_np)
    return Image.fromarray(img_np)

class ClaheTransform:
    def __call__(self,img):
        return apply_Clahe(img)
        
class Rescale:
    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)  # Convert to float
        img_np = img_np / 255.0  # Rescale to [0, 1]
        return Image.fromarray(np.uint8(img_np * 255))

class InvertCXR:
    def __call__(self, img):
        img_np = np.array(img)
        img_np = 255 - img_np
        return Image.fromarray(np.uint8(img_np))
    
# Preprocessing Transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),# Convert to 1 channel
    Rescale(),
    InvertCXR(),
    ClaheTransform(),
    transforms.Resize((img_rows, img_cols)),      # Resize to 256x256
    transforms.ToTensor(),                        # Convert to Tensor
    # transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize to [-1, 1]
])

class XrayDataset(Dataset):
    def __init__(self,source_paths,target_paths,transform=None):
        self.source_paths = source_paths
        self.target_paths = target_paths
        self.transform = transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self,index):
        source_img = Image.open(self.source_paths[index]).convert("L")
        target_img = Image.open(self.target_paths[index]).convert("L")
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)
        return source_img,target_img

source_paths = sorted(glob.glob((os.path.join(source_path,"*.png"))))
target_paths = sorted(glob.glob((os.path.join(target_path,"*.png"))))

source_train, source_val, target_train, target_val = train_test_split(
    source_paths, target_paths, test_size=0.1, random_state=13
)

train_dataset = XrayDataset(source_train,target_train,transform=transform)
val_dataset = XrayDataset(source_val,target_val,transform=transform)

batch_size = 16

train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle = True, num_workers = 4)
val_loader = DataLoader(val_dataset,batch_size = batch_size, shuffle = False, num_workers = 4)

print(f"Train Dataset: {len(train_dataset)} samples.")
print(f"Val dataset: {len(val_dataset)} samples.")

for source,target in train_loader:
    print(f"Source shape: {source[0].shape}, Target shape: {target[0].shape}")
    #print min and max of source and target
    print(f"Source min: {source[0].min()}, Source max: {source[0].max()}")
    break
import matplotlib.pyplot as plt
import torchvision

def visualize_dataloader(train_loader, num_samples=4):
    # Get one batch of data
    data_iter = iter(train_loader)
    source, target = next(data_iter)  
    def unnormalize(tensor):
        return tensor * 0.5 + 0.5  

    # Select few samples to visualize
    source = source[:num_samples]
    target = target[:num_samples]

    # Unnormalize both source and target images
    source = unnormalize(source)
    target = unnormalize(target)

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    fig.suptitle("Source (Original CXR) vs Target (Bone Suppressed CXR)", fontsize=14)

    for i in range(num_samples):
        # Source Image
        axes[i, 0].imshow(source[i][0].cpu().numpy(), cmap="gray")
        axes[i, 0].set_title("Source")
        axes[i, 0].axis("off")

        # Target Image
        axes[i, 1].imshow(target[i][0].cpu().numpy(), cmap="gray")
        axes[i, 1].set_title("Target")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("sample_from_py_file_1024.png")
    # plt.show()

# Call the visualization function
visualize_dataloader(train_loader, num_samples=6)

class ResBlock(nn.Module):
    def __init__(self, filters, scaling=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.scaling = scaling

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        if self.scaling:
            residual = residual * self.scaling
        return x + residual

# Modified architecture with a dilated convolution block
class ResNetBSHighResDilated(nn.Module):
    def __init__(self, img_channels=1, num_filters=64, num_res_blocks=16, res_block_scaling=0.1):
        super(ResNetBSHighResDilated, self).__init__()
        # Downsample 1024x1024 -> 256x256 using a strided convolution
        self.downsample = nn.Conv2d(img_channels, img_channels, kernel_size=3, stride=4, padding=1)
        
        # Optional: a dilated conv block to increase receptive field further
        self.dilated_conv = nn.Conv2d(img_channels, num_filters, kernel_size=3, padding=2, dilation=2)
        
        # Core network
        self.conv_in = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters, res_block_scaling) for _ in range(num_res_blocks)
        ])
        self.conv_mid = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        
        # Upsample back to 1024x1024 (scale factor 4)
        self.upsample = nn.ConvTranspose2d(num_filters, img_channels, kernel_size=4, stride=4, padding=0)
    
    def forward(self, x):
        # x: [B, 1, 1024, 1024]
        x_down = self.downsample(x)       # Now [B, 1, 256, 256]
        x_dilated = self.dilated_conv(x_down)  # Increases receptive field further, output shape [B, num_filters, 256, 256]
        x_in = self.conv_in(x_dilated)
        residual = x_in
        out = x_in
        
        for res_block in self.res_blocks:
            out = res_block(out)
            
        out = self.conv_mid(out)
        out = out + residual
        out = self.conv_out(out)
        out = self.upsample(out)          # Back to [B, 1, 1024, 1024]
        return out

# Log10 Function
def torch_log10(x):
    return torch.log(x) / torch.log(torch.tensor(10.0, device=x.device))

# PSNR Metric
def PSNR(y_true, y_pred, max_pixel=1.0):
    mse = F.mse_loss(y_pred, y_true)
    psnr = 10.0 * torch_log10((max_pixel ** 2) / mse)
    return psnr

# MAE Loss (Mean Absolute Error)
def mae(y_true, y_pred):
    return F.l1_loss(y_pred, y_true)

# SSIM Metric
ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

def ssim(y_true, y_pred):
    return ssim_fn(y_pred, y_true)

# SSIM Loss
def ssim_loss(y_true, y_pred):
    return 1.0 - ssim(y_true, y_pred)

# Multi-Scale SSIM
def ssim_multi(y_true, y_pred):
    return ssim(y_true, y_pred)  

def ssim_multi_loss(y_true, y_pred):
    return 1.0 - ssim(y_true, y_pred)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_highres_dilated = ResNetBSHighResDilated(num_filters=64, num_res_blocks=16, res_block_scaling=0.1).to(device)


class LossMixMulti(nn.Module):
    def __init__(self):
        super(LossMixMulti, self).__init__()
        self.alpha = 0.16
        self.beta = 0.54
        self.gamma = 0.30
        self.ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    def mae(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

    def forward(self, y_true, y_pred):
        # Calculate MAE
        mae_loss = self.mae(y_true, y_pred)

        # Calculate SSIM loss (1 - SSIM to turn into minimization loss)
        ssim_loss = 1 - self.ssim_fn(y_pred, y_true)

        # Calculate PSNR
        psnr_loss = PSNR(y_true, y_pred)

        # Weighted Combination
        loss = self.alpha * mae_loss + self.beta * ssim_loss - self.gamma * PSNR(y_true, y_pred)
        return loss

criterion = LossMixMulti().to(device)

# Optimizer
optimizer = optim.Adam(model_highres_dilated.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5, verbose=True)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                return True
        return False

early_stopping = EarlyStopping(patience=10)


# Directory for saving best model
os.makedirs("weights_1024_invert", exist_ok=True)
best_model_path = "weights_1024_invert/ResNetBS_best.pth"

# Training Loop
n_epoch = 200
n_batch = 16

t = time.time()
best_loss = float("inf")

for epoch in range(n_epoch):
    model_highres_dilated.train()
    epoch_loss = 0.0
    
    print(f"Epoch [{epoch+1}/{n_epoch}]")
    train_loader_tqdm = tqdm(train_loader, desc="Training", leave=True)
    
    # Training Loop
    for source, target in train_loader_tqdm:
        source, target = source.to(device), target.to(device)
        optimizer.zero_grad()
        output = model_highres_dilated(source)
        loss = criterion(target, output)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=loss.item())
    

    epoch_loss /= len(train_loader)
    print(f"Training Loss: {epoch_loss:.4f}")

    # Validation Loop
    model_highres_dilated.eval()
    val_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=True)
    
    with torch.no_grad():
        for source_val, target_val in val_loader_tqdm:
            source_val, target_val = source_val.to(device), target_val.to(device)
            output_val = model_highres_dilated(source_val)
            loss_val = criterion(target_val, output_val)
            val_loss += loss_val.item()
            val_loader_tqdm.set_postfix(val_loss=loss_val.item())
    
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save Best model_highres_dilated
    if val_loss < best_loss:
        print(f"✅ Saving Best model_highres_dilated at Epoch {epoch+1}")
        best_loss = val_loss
        torch.save(model_highres_dilated.state_dict(), best_model_path)
    
    # Reduce LR on Plateau
    scheduler.step(val_loss)

    # Early Stopping
    if early_stopping(val_loss):
        print("🔴 Early Stopping Triggered")
        break

print(f"Total Training Time: {time.time() - t:.2f} seconds")

