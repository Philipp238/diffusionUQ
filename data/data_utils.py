import os
from data.low_dimensional import RegressionDataset
from data.images import CustomImageDataset
import torch
import torchvision
import numpy as np

UCI_DATASET_NAMES = ["concrete", "power-plant", "energy", "kin8nm", "naval-propulsion-plant", "protein-tertiary-structure", "wine-quality-red", "yacht"]

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


def _get_index_train_test_path(data_directory_path, split_num, train = True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return data_directory_path + "index_train_" + str(split_num) + ".txt"
    else:
        return data_directory_path + "index_test_" + str(split_num) + ".txt" 


def get_uci_data(dataset_name, splits=None, standardize=False, validation_ratio=0.0):
    """
    Args:
        dataset_name (str): The name of the UCI dataset
        splits (Union[None, int, List[int]]): None selects all splits, a single int selects a specific split.
        standardize (bool): Whether to transform the feature and labels to have zero mean and unit variance
        validation_ratio (float): The ratio of the validation set w.r.t the training set. Use 0.0 for no validation set.
        
    Return:
        Union(RES, List(RES))
        where RES := (datasets, image_dim, label_dim)
        and datasets is a tuple or triple of the train and test set
                                          or the train, validation and test set
    """
    data_directory_path = os.path.join(
        os.path.dirname(__file__), 
        f"./UCI_Datasets/{dataset_name}/data/"
    )
    
    data = np.loadtxt(f"{data_directory_path}/data.txt")
    index_features = np.loadtxt(f"{data_directory_path}/index_features.txt")
    index_target = np.loadtxt(f"{data_directory_path}/index_target.txt")

    X = data[ : , [int(i) for i in index_features.tolist()] ]
    y = data[ : , [int(index_target.tolist())] ]

    return_singleton = False

    if splits is None:
        splits = range(20)
    elif isinstance(splits, int):
        return_singleton = True
        splits = [splits]

    datasets = []

    for split in splits:
        index_train = np.loadtxt(_get_index_train_test_path(data_directory_path, split, train=True))
        index_test = np.loadtxt(_get_index_train_test_path(data_directory_path, split, train=False))
    
        X_train = X[ [int(i) for i in index_train.tolist()] ]
        y_train = y[ [int(i) for i in index_train.tolist()] ]
        
        X_test = X[ [int(i) for i in index_test.tolist()] ]
        y_test = y[ [int(i) for i in index_test.tolist()] ]
        
        # Convert to tensors
        train_images = torch.tensor(y_train, dtype=torch.float32)
        train_labels = torch.tensor(X_train, dtype=torch.float32)

        test_images = torch.tensor(y_test, dtype=torch.float32)
        test_labels = torch.tensor(X_test, dtype=torch.float32)
        
        if validation_ratio > 0:
            num_training_examples = int(validation_ratio * X_train.shape[0])
            X_val = X_train[num_training_examples:, :]
            y_val = y_train[num_training_examples:]
            X_train = X_train[0:num_training_examples, :]
            y_train = y_train[0:num_training_examples]

            val_images = torch.tensor(y_val, dtype=torch.float32)
            val_labels = torch.tensor(X_val, dtype=torch.float32)

            train_dataset = RegressionDataset(images=train_images, labels=train_labels, standardize=standardize)
            val_dataset = RegressionDataset(images=val_images, labels=val_labels, standardize=standardize)
        else:
            train_dataset = RegressionDataset(images=train_images, labels=train_labels, standardize=standardize)
        
        test_dataset = RegressionDataset(images=test_images, labels=test_labels, standardize=standardize)

        

        image_dim, label_dim = 1, train_labels.shape[1]

        if validation_ratio > 0:
            dataset = (train_dataset, val_dataset, test_dataset),  image_dim, label_dim # type: ignore
        else:
            dataset = (train_dataset, test_dataset),  image_dim, label_dim
        datasets.append(dataset)

    if return_singleton:
        return datasets[0]
    else:
        return datasets
