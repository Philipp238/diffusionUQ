from data.low_dimensional import RegressionDataset
from data.images import CustomImageDataset
import torch
import torchvision

def get_data(dataset_name, dataset_path, dataset_size, standardize=False, image_size=None):
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
        images = torch.tensor(housing['target'], dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(housing['data'], dtype=torch.float32)
        image_dim, label_dim = 1, labels.shape[1]
        
        images = images[:dataset_size]
        labels = labels[:dataset_size]

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