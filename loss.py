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
    return ssim(y_true, y_pred)  # In PyTorch, torchmetrics SSIM is multi-scale by default

def ssim_multi_loss(y_true, y_pred):
    return 1.0 - ssim(y_true, y_pred)



class LossMixMulti084(nn.Module):
    def __init__(self):
        super(LossMixMulti084, self).__init__()
        self.alpha = 0.16
        self.beta = 0.84
        self.ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    def mae(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

    def forward(self, y_true, y_pred):
        # Calculate MAE
        mae_loss = self.mae(y_true, y_pred)

        # Calculate SSIM loss (1 - SSIM to turn into minimization loss)
        ssim_loss = 1 - self.ssim_fn(y_pred, y_true)

        # Weighted Combination
        loss = self.alpha * mae_loss + self.beta * ssim_loss
        return loss
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = LossMixMulti084().to(device)