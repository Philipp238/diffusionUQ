from data.low_dimensional import RegressionDataset
from data.images import CustomImageDataset
import torch
import torchvision

def get_data(dataset_name, dataset_path, dataset_size, standardize=False, image_size=None, logging=None):
    if dataset_name in ['cifar', 'landscape']:
        error_msg = f'Images (datasets with target values that have shape with length > 1) ' + \
                     'are not implemented yet. TODO: image_dim and label_dim have to be shapes then.' + \
                     'Move cifar and landscape back out of the if clause.'
        
        raise NotImplementedError(error_msg)
        
        assert not (image_size is None)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # image_size + 1/4 *image_size
            torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if dataset_name == 'cifar':
            dataset = CustomImageDataset(
                root_dir=dataset_path,  # Path to the dataset.
                transform=transforms,         # Apply the defined transformations.
                image_num=dataset_size               # Limit to 4000 images.
            )
        elif dataset_name == 'landscape':
            dataset = torchvision.datasets.CIFAR10(root=dataset_path, transform=transforms, download=True)
    
    if dataset_name == 'housing_prices':
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        images = torch.tensor(housing['target'], dtype=torch.float32).unsqueeze(-1)  # (20640, 1)
        labels = torch.tensor(housing['data'], dtype=torch.float32) # (20640, 8)
        image_dim, label_dim = 1, labels.shape[1]
        
        images = images[:dataset_size]
        labels = labels[:dataset_size]

        dataset = RegressionDataset(images=images, labels=labels, standardize=standardize)
    elif dataset_name == 'concrete':
        from ucimlrepo import fetch_ucirepo 
  
        # fetch dataset 
        yacht_hydrodynamics = fetch_ucirepo(id=165) # Concrete Compressive Strength dataset (https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
        
        # data (as pandas dataframes) 
        features = yacht_hydrodynamics.data.features 
        targets = yacht_hydrodynamics.data.targets

        # Convert to tensors
        images = torch.tensor(targets.values, dtype=torch.float32)  # (N, 1)
        labels = torch.tensor(features.values, dtype=torch.float32) # (N, 8)

        # Image and label dimensions
        image_dim, label_dim = 1, labels.shape[1]

        # Optional: select a subset
        dataset_size = len(labels)  # or set to a smaller number like 1000
        images = images[:dataset_size]
        labels = labels[:dataset_size]

        # Example flag for standardization (if your RegressionDataset uses this)
        standardize = True

        # Assume you have a custom RegressionDataset class
        dataset = RegressionDataset(images=images, labels=labels, standardize=standardize)
    elif dataset_name == 'power':
        from ucimlrepo import fetch_ucirepo 
  
        # fetch dataset 
        combined_cycle_power_plant = fetch_ucirepo(id=294) # Yacht Hydrodynamics (https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics)
        
        # data (as pandas dataframes) 
        features = combined_cycle_power_plant.data.features 
        targets = combined_cycle_power_plant.data.targets

        # Convert to tensors
        images = torch.tensor(targets.values, dtype=torch.float32)  # (N, 1)
        labels = torch.tensor(features.values, dtype=torch.float32) # (N, 8)

        # Image and label dimensions
        image_dim, label_dim = 1, labels.shape[1]

        # Optional: select a subset
        dataset_size = len(labels)  # or set to a smaller number like 1000
        images = images[:dataset_size]
        labels = labels[:dataset_size]

        # Example flag for standardization (if your RegressionDataset uses this)
        standardize = True

        # Assume you have a custom RegressionDataset class
        dataset = RegressionDataset(images=images, labels=labels, standardize=standardize)
        
    elif dataset_name == 'energy':
        raise NotImplementedError(f'The energy dataset has 2 target variables and it seems like our framework has to be adapted for this first.')
        
        from ucimlrepo import fetch_ucirepo 
  
        # fetch dataset 
        energy_efficiency = fetch_ucirepo(id=242) 
        
        # data (as pandas dataframes) 
        features = energy_efficiency.data.features 
        targets = energy_efficiency.data.targets 
        
        # Convert to tensors
        images = torch.tensor(targets.values, dtype=torch.float32)  # (N, 1)
        labels = torch.tensor(features.values, dtype=torch.float32) # (N, 8)

        # Image and label dimensions
        image_dim, label_dim = 1, labels.shape[1]

        # Optional: select a subset
        dataset_size = len(labels)  # or set to a smaller number like 1000
        images = images[:dataset_size]
        labels = labels[:dataset_size]

        # Example flag for standardization (if your RegressionDataset uses this)
        standardize = True

        # Assume you have a custom RegressionDataset class
        dataset = RegressionDataset(images=images, labels=labels, standardize=standardize)
    elif dataset_name == 'x-squared':
        image_dim, label_dim = 1, 1
        x = (torch.rand(dataset_size, dtype=torch.float32) * 3)
        y = x**2
        dataset = RegressionDataset(images=y.unsqueeze(-1), labels=x.unsqueeze(-1), standardize=standardize)
    elif dataset_name == 'uniform-regression':
        labels = torch.rand(dataset_size, dtype=torch.float32) * 5
        images = torch.rand(dataset_size, dtype=torch.float32) + labels
        
        dataset = RegressionDataset(images=images.unsqueeze(-1), labels=labels.unsqueeze(-1), standardize=standardize)
        label_dim = 1
        image_dim = 1

    return dataset, image_dim, label_dim