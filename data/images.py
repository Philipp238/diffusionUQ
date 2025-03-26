import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split

# Define the Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_num=4000):
        # Collects image file paths from the root directory, limited to `image_num` images.
        self.image_paths = sorted(
            [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
             if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )[:image_num]  # Limit to the first `image_num` images
        self.transform = transform # Transformation to apply to images

    def __len__(self):
        # Returns the number of images in the dataset.
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Loads an image by index, converts it to RGB, and applies transformations if provided.
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Since no actual labels, return 0 as dummy labels
        return image, 0

    def set_transform(self, transform):
        # Allows dynamic updating of transformations after dataset initialization.
        self.transform = transform


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    
    
