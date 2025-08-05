"""Script to download, process and split Darcy Flow datasets from the PDEBench source."""

import os
import numpy as np
import xarray as xr
from torchvision.datasets.utils import download_url

# Dictionary with datasets and urls
DATASETS = {
    "1D_Burgers": "https://darus.uni-stuttgart.de/api/access/datafile/281363",
}


def data_split(ds, train_test_split, filename, max_steps = 104):
    """Split darcy flow dataset into training and testing datasets

    Args:
        ds (_type_): Input dataset
        seed (int, optional): Random seed.
        train_size (float, optional): Size of training dataset.

    Returns:
        _type_: Output training and testing datasets
    """

    if filename.startswith("1D"):
        ds = ds.isel(phony_dim_2 = slice(0,max_steps), phony_dim_0 = slice(0,max_steps))
        # t_reduced = ds["t-coordinate"][:-1]
        # ds = ds.drop_vars("t-coordinate")
        # ds["t-coordinate"] = t_reduced 
        ds = ds.rename(
            {
                "phony_dim_0": "t",
                "phony_dim_1": "samples",
                "phony_dim_2": "t",
                "phony_dim_3": "x",
            }
        ).rename_vars({"tensor": "u"})
    n_samples = ds.sizes["samples"]

    # Train test split
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * train_test_split)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Create data
    train_data = ds.isel(samples = train_indices)
    test_data = ds.isel(samples = test_indices)

    # Save standardization constants as attributes
    mean = train_data.u.mean().item()
    std = train_data.u.std().item()

    train_data.attrs["mean"] = mean
    train_data.attrs["std"] = std
    test_data.attrs["mean"] = mean
    test_data.attrs["std"] = std

    return train_data, test_data


def main(data_directory, name, url, train_test_split, download=True, remove=True):
    """Main function to download, process and split Darcy Flow datasets

    Args:
        data_dir (_type_): Data directory
        train_split (_type_): Train test split ratio
        seed (_type_): Random seed
        download (bool, optional): Whether to download data. Defaults to True.
        remove (bool, optional): Whether to remove raw data. Defaults to True.
    """
    data_dir = data_directory + f"{name}/"
    filename = f"{name}.hdf5"
    if download:
        file_path = os.path.join(data_dir + "raw/")
        download_url(url, file_path, filename=filename)
    # Create output folder
    os.makedirs(data_dir + "processed/", exist_ok=True)
    # Load datasets and create train/test splits
    file_path = data_dir + f"raw/{filename}"
    ds = xr.load_dataset(file_path)
    train_data, test_data = data_split(ds, train_test_split, filename)

    train_data.to_netcdf(data_dir + "processed/train.nc")
    test_data.to_netcdf(data_dir + "processed/test.nc")

    # Remove raw data
    if remove:
        os.system(f"rm -r {data_dir}/raw")


if __name__ == "__main__":
    train_test_split = 0.9
    seed = 42
    np.random.seed(seed)
    download = False
    remove = False
    data_dir = "data/"
    for dataset in DATASETS:
        print(f"Downloading {dataset} from {DATASETS[dataset]}")
        main(data_dir, dataset, DATASETS[dataset], train_test_split, download, remove)
