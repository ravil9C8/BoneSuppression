import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import warnings 
warnings.filterwarnings('ignore',category=FutureWarning) #because of numpy version
import re
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import threading
import gc

from models import ResNetBSHighResDilated
from utils import process_image_bs
MODEL_ROOT = "/mnt/nvme_disk2/User_data/rp926k/Bone_Supression/weights_1024_pyscript_not_norm"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class ModelContainer:
    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.local_data = threading.local()

    def _load_model(self, model_key,path):
        if model_key=="bone_supression_model":
            model = ResNetBSHighResDilated(num_filters=64, num_res_blocks=16, res_block_scaling=0.1)
            model.load_state_dict(torch.load(path,weights_only=True))
            model.eval()
            return model.to(device)
        else:
            raise ValueError(f"Model key {model_key} not recognized")
        
    def get_model(self, model_key):
        if not hasattr(self.local_data, "models"):
            self.local_data.models = {}
        if model_key not in self.local_data.models:
            self.local_data.models[model_key] = self._load_model(model_key,self.model_paths[model_key])
        return self.local_data.models[model_key]
    

model_paths = {
    'bone_supression_model': os.path.join(MODEL_ROOT, "ResNetBS_best.pth")
}

model_container = ModelContainer(model_paths)

def cleanup_gpu_memory():
    """Cleanup GPU memory by emptying the cache."""
    gc.collect()
    torch.cuda.empty_cache()

def get_bone_supressed_resnet(input_image: torch.Tensor, is_inverted: bool) -> np.ndarray:
    '''
    Input: 
    input_image: torch.Tensor of shape [1, 3, 1024, 1024]
    is_inverted: bool, if True, the input image is inverted

    Output:
    bs_image: np.ndarray of shape [1,1024,1024]
    '''
    bone_supression_model = model_container.get_model("bone_supression_model")
    bone_supression_model.eval()

    # Preprocess the input image
    input_image = process_image_bs(input_image)
    print("Max pixel value in the input image:", torch.max(input_image))
    print("Min pixel value in the input image:", torch.min(input_image))
    if is_inverted:
        input_image = 1 - input_image
    with torch.no_grad():
        bs_image = bone_supression_model(input_image.to(device))
    bs_image = bs_image.squeeze().cpu().numpy()
    bs_image = np.clip(bs_image, 0, 1)
    #Return the bone supressed image by changing shape of it from [1024,1024] to [1024,1024,3]
    bs_image = np.repeat(bs_image[:, :, np.newaxis], 3, axis=2)

    return 1 - bs_image

if __name__ == "__main__":
    input_image_path = "/mhgp003-v1/kanpur/data_radiovision/validation_dataset/Validation_Dataset_2/PTCXR (1282).png"
    image_ = cv2.imread(input_image_path)
    image_ = cv2.resize(image_, (1024, 1024))
    print("Input Image Shape:", image_.shape)
    image_ = np.transpose(image_, (2, 0, 1))  # Shape: (3, 1024, 1024)
    # resize the image to (1024, 1024, 3)
    
    
    print("Input Image Shape:", image_.shape)
    image_ = np.expand_dims(image_, axis=0)
    print("Max pixel value in the input image:", np.max(image_))
    print("Min pixel value in the input image:", np.min(image_))
    tensor_image = torch.from_numpy(image_)

    bs_image = get_bone_supressed_resnet(tensor_image, is_inverted=True)
    print("Max pixel value in the bone supressed image:", np.max(bs_image))
    print("Min pixel value in the bone supressed image:", np.min(bs_image))
    # print("Max pixel value in the bone supressed image:", np.max(original_image_bs))
    # print("Min pixel value in the bone supressed image:", np.min(original_image_bs))
    print(bs_image.shape)
    bs_image_uint8 = (bs_image * 255).astype(np.uint8)
    # bone_supressed = (bone_supressed * 255).astype(np.uint8)
    cv2.imwrite("bone_supressed_np_test_image.png", bs_image_uint8)
    plt.imshow(bs_image, cmap="gray")
    plt.axis("off")
    plt.savefig("bone_supressed.png")