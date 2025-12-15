# Includes the autoregressive datasets, i.e. one-dimensional PDEs and WeatherBench dataset.

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

WB_INPUT = [
    "2m_temperature",
    "temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential",
    "land_sea_mask",
    "geopotential_at_surface",
]
WB_STATIC = ["land_sea_mask", "geopotential_at_surface"]
WB_DYNAMIC = [
    "2m_temperature",
    "temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential",
]
WB_TARGET = ["2m_temperature"]  # Idx of input variables to use as target


class PDE1D(Dataset):
    """
    A class used to handle one-dimensional PDE datasets from PDEBench.
    Shape of PDE1D dataset: (samples, time, x)
    """

    def __init__(
        self,
        data_dir: str,
        pde: str,
        var: str = "train",
        downscaling_factor: int = 1,
        temporal_downscaling_factor: int = 2,
        last_t_steps: int = 2,
        normalize=True,
        select_timesteps: str = "zero",
        seed: int = 0,
        train_split: float = 0.9,
    ) -> None:
        """Initialize the PDE dataset.

        Args:
            data_dir (str): Data directory
            pde (str): PDE choice.
            var (str, optional): Train/test/val variable. Defaults to "train".
            downscaling_factor (int, optional): Grid downscaling factor. Defaults to 1.
            temporal_downscaling_factor (int, optional): Temporal downscaling factor. Defaults to 2.
            last_t_steps (int, optional): Last t timesteps to use as input. Defaults to 2.
            normalize (bool, optional): Normalization. Defaults to True.
            select_timesteps (str, optional): Timestep selection method. Defaults to "zero".
            seed (int, optional): Seed. Defaults to 0.
            train_split (float, optional): Train-test-val split. Defaults to 0.9.

        Raises:
            ValueError: Scaling factors must be integers.
        """

        if pde not in ["Advection", "Burgers", "ReacDiff", "KS"]:
            raise ValueError(
                "PDE must be one of the following: 'Advection', 'Burgers', 'ReacDiff', 'KS'"
            )
        self.pde = pde
        self.var = var
        self.train_split = train_split
        self.seed = seed
        self.filename = "test.nc" if var == "test" else "train.nc"
        if var == "ood":
            self.filename = f"{var}.nc"
        self.path = os.path.join(data_dir, f"1D_{self.pde}/processed/")
        self.dataset = xr.open_dataset(self.path + self.filename)
        assert isinstance(downscaling_factor, int), "Scaling factor must be Integer"
        self.downscaling_factor = downscaling_factor
        assert isinstance(temporal_downscaling_factor, int), (
            "Temporal scaling factor must be Integer"
        )
        self.temporal_downscaling_factor = temporal_downscaling_factor
        self.normalize = normalize
        self.data_dir = data_dir
        self.select_timesteps = select_timesteps
        self.last_t_steps = last_t_steps

        # Get normalization
        self.mean = self.dataset.attrs.get("mean", None)
        self.std = self.dataset.attrs.get("std", None)

        # Get sizes
        self.n = len(self.dataset["samples"])
        self.t_size = (
            len(self.dataset["t-coordinate"][:: self.temporal_downscaling_factor])
            - self.last_t_steps
        )

        # Get data
        index = self.train_val_split()
        if index is not None:
            self.data = self.dataset.u[
                index, :: self.temporal_downscaling_factor, :: self.downscaling_factor
            ].load()
            self.n = len(index)
        else:
            self.data = self.dataset.u[
                :, :: self.temporal_downscaling_factor, :: self.downscaling_factor
            ].load()
        self.grid = self.dataset["x-coordinate"][
            :: self.downscaling_factor
        ].values.copy()

    def train_val_split(self):
        rng = np.random.default_rng(seed=self.seed)
        indices = rng.choice(self.n, size=self.n, replace=False)
        n_train = int(self.n * self.train_split)
        if self.var == "train":
            return indices[:n_train]
        elif self.var == "val":
            return indices[n_train:]
        else:
            return None

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        if self.select_timesteps == "zero" or self.select_timesteps == "random":
            return self.n
        elif self.select_timesteps == "all":
            return self.n * self.t_size

    def __getitem__(self, idx: int) -> tuple:
        """Returns the idx-th element of the dataset

        Args:
            idx (int): Index of the element to be returned

        Returns:
            tuple: Tuple containing the input and output tensors
        """

        if self.select_timesteps == "zero":
            time_idx = 0
            sample_idx = idx
        elif self.select_timesteps == "random":
            time_idx = torch.randint(low=0, high=self.t_size, size=(1,)).item()
            sample_idx = idx
        elif self.select_timesteps == "all":
            sample_idx = idx // self.t_size
            time_idx = idx % (self.t_size) if sample_idx > 0 else idx
        else:
            raise ValueError("select_timesteps must be 'zero', 'random', or 'all'")

        input = self.data[
            sample_idx, time_idx : time_idx + self.last_t_steps
        ].values.copy()
        target = self.data[sample_idx, time_idx + self.last_t_steps].values.copy()

        # Normalize
        if self.normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std

        # Get residual
        residual = target - input[-1]

        input_tensor = torch.cat(
            [
                torch.tensor(input, dtype=torch.float32),
                torch.tensor(self.grid, dtype=torch.float32).unsqueeze(0),
            ],
            dim=0,
        )
        target_tensor = torch.tensor(residual, dtype=torch.float32).unsqueeze(0)

        return target_tensor, input_tensor

    def get_trajectory(self, idx: int, length: int = 10) -> torch.Tensor:
        """Returns a trajectory of the idx-th element of the dataset starting from zero

        Args:
            idx (int): Index of the element to be returned
            length (int, optional): Length of the trajectory. Defaults to 10.

        Returns:
            tuple: Trajectory tensor, Input tensor
        """
        input = self.data[idx, 0 : self.last_t_steps].values.copy()
        target = self.data[idx, 0 : (self.last_t_steps + length)].values.copy()

        # Normalize
        if self.normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std

        input_tensor = torch.cat(
            [
                torch.tensor(input, dtype=torch.float32),
                torch.tensor(self.grid, dtype=torch.float32).unsqueeze(0),
            ],
            dim=0,
        )
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        return target_tensor, input_tensor

    def get_coordinates(self) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset["x-coordinate"][:: self.downscaling_factor].values
        return (x,)

    def get_grid(self) -> Tuple:
        """Returns the grid of the dataset.

        Returns:
            Tuple: Tuple containing x and t coordinate values
        """
        x = self.dataset["x-coordinate"][:: self.downscaling_factor].values
        t = self.dataset["t-coordinate"][:: self.temporal_downscaling_factor].values
        return (x, t)

    def get_dimensions(self) -> Tuple:
        """Get dimensions of the coordinates.

        Returns:
            Tuple: Size of the data-dimensions.
        """
        x = len(self.dataset["x-coordinate"][:: self.downscaling_factor].values)
        return (x,)

    def destandardize_output(self, u:torch.Tensor)->torch.Tensor:
        """Destandardize a tensor.

        Args:
            u (torch.Tensor): Input Tensor.

        Returns:
            torch.Tensor: Destandardized output tensor
        """
        if hasattr(self, "mean") and hasattr(self, "std"):
            return u * self.std + self.mean
        else:
            return u


class WeatherBench(Dataset):
    """
    A class used to handle the WeatherBench dataset, downloaded using the corresponding script.
    """

    def __init__(
        self,
        data_path: str = "/home/groups/ai/datasets/weather_forecasting/",
        var: str = "train",
        normalize=True,
        downscaling_factor: int = 1,
        last_t_steps: int = 2,
        preload: bool = True,
    ):
        """Initialize WeatherBench2 dataset.

        Args:
            data_path (str, optional): Data path. Defaults to "/home/groups/ai/datasets/weather_forecasting/".
            var (str, optional): Train/test/val selection. Defaults to "train".
            normalize (bool, optional): Normalization. Defaults to True.
            downscaling_factor (int, optional): Grid downscaling factor. Defaults to 1.
            last_t_steps (int, optional): Last t timesteps to use. Defaults to 2.
            preload (bool, optional): Whether to preload the whole dataset into RAM. Defaults to True.
        """
        self.var = var
        self.normalize = normalize
        self.last_t_steps = last_t_steps
        self.downscaling_factor = downscaling_factor

        time_slice = self.get_split(self.var)
        zarr_path = os.path.join(data_path, "era5.zarr")
        self.dataset = xr.open_zarr(zarr_path, consolidated=False)[WB_INPUT].sel(
            time=time_slice
        )
        self.mean, self.std = self.get_statistics(data_path)
        self.dataset = self.dataset.isel(
            latitude=slice(None, None, downscaling_factor),
            longitude=slice(None, None, downscaling_factor),
        )
        if preload:
            self.data_array = self.dataset.to_array().load()
        else:
            self.data_array = None

        self.time_len = self.dataset.time[4:-last_t_steps][::4].size  # Adjust for UTC00
        self.n_vars = len(WB_INPUT)

    @staticmethod
    def get_split(var: str) -> slice:
        if var == "train":
            return slice("2010", "2019")  # Change first argument to 2010
        elif var == "val":
            return slice("2020", "2020")
        elif var == "test":
            return slice("2021", "2022")
        else:
            raise AssertionError(f"{var} is not in [train, val, test]")

    @staticmethod
    def get_statistics(data_path: str) -> Tuple:
        """Load data statistics from path.

        Args:
            data_path (str): Path to data folder

        Returns:
            Tuple: Tuple of mean and standard deviation of data.
        """
        statistics = np.load(os.path.join(data_path, "era5_statistics.npy"))
        mean, std = np.split(statistics, 2, axis=-1)
        return mean.squeeze(), std.squeeze()

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return self.time_len

    def __getitem__(self, idx: int) -> tuple:
        """Returns the idx-th element of the dataset

        Args:
            idx (int): Index of the element to be returned

        Returns:
            tuple: Tuple containing the input and output tensors
        """
        # Create index for UTC00 and 6h forecast
        utc_idx = ((idx + 1) * 4) - self.last_t_steps + 1
        if self.data_array is not None:
            sample = self.data_array[
                :, utc_idx : utc_idx + self.last_t_steps + 1
            ].values.copy()
        else:
            sample = (
                self.dataset.isel(time=slice(utc_idx, utc_idx + self.last_t_steps + 1))
                .to_array()
                .values.copy()
            )

        # Shape: [variables, time, lat, lon]
        if self.normalize:
            sample = (sample - self.mean[:, None, None, None]) / self.std[
                :, None, None, None
            ]

        # Input: [dynamic (all but WB_STATIC) over last_t_steps] + [static (first time)]
        input_dynamic = sample[np.isin(WB_INPUT, WB_DYNAMIC)][:, : self.last_t_steps]
        input_dynamic = input_dynamic.reshape(
            -1, *input_dynamic.shape[2:]
        )  # flatten time and var dims

        static = sample[np.isin(WB_INPUT, WB_STATIC), 0]  # static at t=0
        input_tensor = torch.from_numpy(
            np.concatenate([input_dynamic, static], axis=0)
        ).float()

        # Target: delta between last and previous timestep for WB_TARGET
        target_idx = np.isin(WB_INPUT, WB_TARGET)
        target_t = sample[target_idx, self.last_t_steps]
        target_t_prev = sample[target_idx, self.last_t_steps - 1]
        target_tensor = torch.from_numpy(target_t - target_t_prev).float()

        return target_tensor, input_tensor

    def destandardize_output(self, u:torch.Tensor)->torch.Tensor:
        """Destandardize a tensor.

        Args:
            u (torch.Tensor): Input Tensor.

        Returns:
            torch.Tensor: Destandardized output tensor
        """
        if self.normalize:
            target_idx = np.isin(WB_INPUT, WB_TARGET)
            return (
                u * self.std[target_idx].squeeze().item()
                + self.mean[target_idx].squeeze().item()
            )
        else:
            return u

    def get_grid(self) -> Tuple:
        """Returns the grid of the dataset.

        Returns:
            Tuple: Tuple containing x (lat,lon) and t coordinate values
        """
        lat = self.dataset.latitude.values
        lon = self.dataset.longitude.values
        t = self.dataset.time[4:][::4] + pd.Timedelta("6h")  # Returns prediction time
        return ((lat, lon), t)

    def get_dimensions(self) -> Tuple:
        """Get dimensions of the coordinates.

        Returns:
            Tuple: Size of the data-dimensions (lat,lon).
        """
        x = self.dataset.sizes["latitude"]
        y = self.dataset.sizes["longitude"]
        return (x, y)


if __name__ == "__main__":
    # # Example usage
    dataset = WeatherBench(
        var="train", normalize=True, downscaling_factor=2, preload=False
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Temporal: {dataset.time_len}")
    target_tensor, input_tensor = dataset.__getitem__(728)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Target tensor shape: {target_tensor.shape}")

    ll, t = dataset.get_grid()
    lat, lon = ll
    print(lat.shape, lon.shape, t.shape)
    x, y = dataset.get_dimensions()
    print(x, y)