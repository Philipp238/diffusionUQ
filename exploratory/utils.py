import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


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
        
class RegressionDataset(Dataset):
    def __init__(self, images, labels, standardize=True):
        # names (images, labels) taken from conditional image generation, to make the roles more clear (images=target (y), labels=conditioning (x))
        super().__init__()
        if standardize:
            self.images_means = images.mean(dim=0, keepdim=True)
            self.images_stds = images.std(dim=0, keepdim=True)
            images = (images - self.images_means) / self.images_stds
            
            self.labels_mean = labels.mean()
            self.labels_std = labels.std()
            labels = (labels - self.labels_mean) / self.labels_std
            
        self.images = images
        self.labels = labels
        
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
        
    def __len__(self):
        return len(self.labels)

class LowDimensionalGenerationDataset(Dataset):
    def __init__(self, x):
        super().__init__()
        self.x = x
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index]


def get_data(args):
    if args.dataset_name in ['cifar', 'landscape']:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    if args.dataset_name == 'cifar':
        dataset = CustomImageDataset(
            root_dir=args.dataset_path,  # Path to the dataset.
            transform=transforms,         # Apply the defined transformations.
            image_num=args.image_num               # Limit to 4000 images.
        )
    elif args.dataset_name == 'landscape':
        dataset = torchvision.datasets.CIFAR10(root=args.dataset_path, transform=transforms, download=True)
    elif args.dataset_name == 'housing_prices':
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        images = torch.tensor(housing['target'], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(housing['data'], dtype=torch.float32)
        image_dim, label_dim = 1, labels.shape[1]

        dataset = RegressionDataset(images=images, labels=labels, standardize=True)
    elif args.dataset_name == 'x_squared':
        image_dim, label_dim = 1, 1
        x = (torch.randn(3, dtype=torch.float32) * 3)
        y = x**2
        dataset = RegressionDataset(images=x.unsqueeze(-1), labels=y, standardize=False)
    elif args.dataset_name == 'uniform':
        x = torch.rand(args.dataset_size, dtype=torch.float32)
        dataset = LowDimensionalGenerationDataset(x.unsqueeze(-1))
    elif args.dataset_name == 'uniform-regression':
        labels = torch.randn(args.dataset_size, dtype=torch.float32) * 5
        images = torch.rand(args.dataset_size, dtype=torch.float32) + labels  ** 2
        
        dataset = RegressionDataset(images=images.unsqueeze(-1), labels=labels.unsqueeze(-1), standardize=False)
        label_dim = 1
        image_dim = 1
    elif args.dataset_name == 'uniform-conditioning':
        labels = torch.randint(args.n_classes, (args.dataset_size, ), dtype=torch.float32)
        images = torch.rand(args.dataset_size, dtype=torch.float32) + labels * 3
        
        dataset = RegressionDataset(images=images.unsqueeze(-1), labels=labels.unsqueeze(-1), standardize=False)
        
    if args.dataset_name in ['cifar', 'landscape', 'uniform', 'uniform-conditioning']: # generative tasks
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        return dataloader
    else:  # regression tasks
        training_dataset, validation_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
        training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size_pred, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_pred, shuffle=True)
        
        return training_dataloader, validation_dataloader, test_dataloader, image_dim, label_dim


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    
    
