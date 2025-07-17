import os

import numpy as np
import pandas as pd
import torch
import torchvision

from data.images import CustomImageDataset
from data.low_dimensional import RegressionDataset
from data.datasets import PDE1D, PDE2D

UCI_DATASET_NAMES = [
    "concrete",
    "power-plant",
    "energy",
    "kin8nm",
    "naval-propulsion-plant",
    "protein-tertiary-structure",
    "wine-quality-red",
    "yacht",
]


def get_data(
    dataset_name,
    dataset_path,
    data_parameters,
    logging=None,
):
    standardize = data_parameters["standardize"]
    select_timesteps = data_parameters["select_timesteps"]
    temporal_downscaling_factor = data_parameters["temporal_downscaling_factor"]
    if dataset_name in ["1D_Advection", "1D_ReacDiff", "1D_Burgers", "1D_KS"]:
        train_dataset = PDE1D(
            data_dir=dataset_path,
            pde=dataset_name.split("_")[1],
            var="train",
            downscaling_factor=data_parameters["downscaling_factor"],
            normalize=standardize,
            select_timesteps=select_timesteps,
            temporal_downscaling_factor=temporal_downscaling_factor,
        )
        val_dataset = PDE1D(
            data_dir=dataset_path,
            pde=dataset_name.split("_")[1],
            var="val",
            downscaling_factor=data_parameters["downscaling_factor"],
            normalize=standardize,
            select_timesteps=select_timesteps,
            temporal_downscaling_factor=temporal_downscaling_factor,
        )
        test_dataset = PDE1D(
            data_dir=dataset_path,
            pde=dataset_name.split("_")[1],
            var="test",
            downscaling_factor=data_parameters["downscaling_factor"],
            normalize=standardize,
            select_timesteps=select_timesteps,
            temporal_downscaling_factor=temporal_downscaling_factor,
        )
    elif dataset_name in ["2D_DarcyFlow"]:
        train_dataset = PDE2D(
            data_dir=dataset_path,
            pde=dataset_name.split("_")[1],
            var="train",
            downscaling_factor=data_parameters["downscaling_factor"],
            normalize=standardize,
        )
        val_dataset = PDE2D(
            data_dir=dataset_path,
            pde=dataset_name.split("_")[1],
            var="val",
            downscaling_factor=data_parameters["downscaling_factor"],
            normalize=standardize,
        )
        test_dataset = PDE2D(
            data_dir=dataset_path,
            pde=dataset_name.split("_")[1],
            var="test",
            downscaling_factor=data_parameters["downscaling_factor"],
            normalize=standardize,
        )
    dataset = (train_dataset, val_dataset,test_dataset)
    target_dim, input_dim = (
        (1, *train_dataset.get_dimensions()),
        (3, *train_dataset.get_dimensions()),
    )

    return dataset, target_dim, input_dim


def _get_index_train_test_path(data_directory_path, split_num, train=True):
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


def _onehot_encode_cat_feature(X, cat_var_idx_list):
    """
    Apply one-hot encoding to the categorical variable(s) in the feature set,
        specified by the index list.
    """
    # select numerical features
    X_num = np.delete(arr=X, obj=cat_var_idx_list, axis=1)
    # select categorical features
    X_cat = X[:, cat_var_idx_list]
    X_onehot_cat = []
    for col in range(X_cat.shape[1]):
        X_onehot_cat.append(pd.get_dummies(X_cat[:, col], drop_first=True))
    X_onehot_cat = np.concatenate(X_onehot_cat, axis=1).astype(np.float32)
    dim_cat = X_onehot_cat.shape[1]  # number of categorical feature(s)
    X = np.concatenate([X_num, X_onehot_cat], axis=1)
    return X, dim_cat


def _preprocess_uci_feature_set(X, dataset_name):
    """
    Obtain preprocessed UCI feature set X (one-hot encoding applied for categorical variable)
        and dimension of one-hot encoded categorical variables.
    """
    dim_cat = 0
    if dataset_name == "bostonHousing":
        X, dim_cat = _onehot_encode_cat_feature(X, [3])
    elif dataset_name == "energy":
        X, dim_cat = _onehot_encode_cat_feature(X, [4, 6, 7])
    elif dataset_name == "naval-propulsion-plant":
        X, dim_cat = _onehot_encode_cat_feature(X, [0, 1, 8, 11])
    else:
        pass

    return X, dim_cat


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
        os.path.dirname(__file__), f"./UCI_Datasets/{dataset_name}/data/"
    )

    data = np.loadtxt(f"{data_directory_path}/data.txt")
    index_features = np.loadtxt(f"{data_directory_path}/index_features.txt")
    index_target = np.loadtxt(f"{data_directory_path}/index_target.txt")

    X = data[:, [int(i) for i in index_features.tolist()]]
    y = data[:, [int(index_target.tolist())]]

    # preprocess feature set X
    X, dim_cat = _preprocess_uci_feature_set(X=X, dataset_name=dataset_name)

    return_singleton = False

    if splits is None:
        splits = range(20)
    elif isinstance(splits, int):
        return_singleton = True
        splits = [splits]

    datasets = []

    for split in splits:
        index_train = np.loadtxt(
            _get_index_train_test_path(data_directory_path, split, train=True)
        )
        index_test = np.loadtxt(
            _get_index_train_test_path(data_directory_path, split, train=False)
        )

        X_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]]

        X_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]]

        # Convert to tensors
        train_images = torch.tensor(y_train, dtype=torch.float32)
        train_labels = torch.tensor(X_train, dtype=torch.float32)

        test_images = torch.tensor(y_test, dtype=torch.float32)
        test_labels = torch.tensor(X_test, dtype=torch.float32)

        if validation_ratio > 0:
            num_training_examples = int((1 - validation_ratio) * X_train.shape[0])
            X_val = X_train[num_training_examples:, :]
            y_val = y_train[num_training_examples:]
            X_train_subset = X_train[0:num_training_examples, :]
            y_train_subset = y_train[0:num_training_examples]

            val_images = torch.tensor(y_val, dtype=torch.float32)
            val_labels = torch.tensor(X_val, dtype=torch.float32)

            train_images_subset = torch.tensor(y_train_subset, dtype=torch.float32)
            train_labels_subset = torch.tensor(X_train_subset, dtype=torch.float32)

            train_dataset = RegressionDataset(
                target=train_images_subset,
                input=train_labels_subset,
                standardize=standardize,
                dim_cat=dim_cat,
            )
            train_std_params = train_dataset.get_std_params()
            val_dataset = RegressionDataset(
                target=val_images,
                input=val_labels,
                standardize=standardize,
                std_params=train_std_params,
                dim_cat=dim_cat,
            )
        else:
            train_dataset = RegressionDataset(
                target=train_images,
                input=train_labels,
                standardize=standardize,
                dim_cat=dim_cat,
            )
            train_std_params = train_dataset.get_std_params()

        test_dataset = RegressionDataset(
            target=test_images,
            input=test_labels,
            standardize=standardize,
            std_params=train_std_params,
            dim_cat=dim_cat,
        )

        image_dim, label_dim = (1,1), (1,train_labels.shape[1])

        if validation_ratio > 0:
            dataset = (train_dataset, val_dataset, test_dataset), image_dim, label_dim  # type: ignore
        else:
            dataset = (train_dataset, test_dataset), image_dim, label_dim
        datasets.append(dataset)

    if return_singleton:
        return datasets[0]
    else:
        return datasets
