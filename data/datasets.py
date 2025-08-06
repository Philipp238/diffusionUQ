import os
from typing import Tuple

import torch
import xarray as xr
from torch.utils.data import Dataset
import numpy as np

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
        """Initialize PDE1D Dataset

        Args:
            data_dir (str): Data directory.
            test (bool, optional): Whether to load train or test data. Defaults to False.
            downscaling_factor (int, optional): Downscaling for spatial resolution. Defaults to 1.
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
        rng = np.random.default_rng(seed = self.seed)
        indices = rng.choice(self.n, size = self.n, replace = False)
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
            torch.Tensor: Trajectory tensor
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
        x = self.dataset["x-coordinate"][:: self.downscaling_factor].values
        t = self.dataset["t-coordinate"][:: self.temporal_downscaling_factor].values
        return (x, t)

    def get_dimensions(self) -> Tuple:
        x = len(self.dataset["x-coordinate"][:: self.downscaling_factor].values)
        return (x,)

    def destandardize_output(self, u):
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
           
        self.time_len = self.dataset.sizes["time"] - last_t_steps
        self.n_vars = len(WB_INPUT)

    @staticmethod
    def get_split(var: str) -> slice:
        if var == "train":
            return slice("2019", "2019")  # Change first argument to 2010
        elif var == "val":
            return slice("2020", "2020")
        elif var == "test":
            return slice("2021", "2022")
        else:
            raise AssertionError(f"{var} is not in [train, val, test]")

    @staticmethod
    def get_statistics(data_path: str) -> Tuple:
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
        if self.data_array is not None:
            sample = self.data_array[:, idx : idx + self.last_t_steps + 1].values.copy()
        else:
            sample = (
                self.dataset.isel(time=slice(idx, idx + self.last_t_steps + 1))
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

    def destandardize_output(self, u):
        if self.normalize:
            target_idx = np.isin(WB_INPUT, WB_TARGET)
            return (
                u * self.std[target_idx].squeeze().item()
                + self.mean[target_idx].squeeze().item()
            )
        else:
            return u

    def get_grid(self) -> Tuple:
        lat = self.dataset.latitude.values
        lon = self.dataset.longitude.values
        t = self.dataset.time.values
        return ((lat, lon), t)

    def get_dimensions(self) -> Tuple:
        x = self.dataset.sizes["latitude"]
        y = self.dataset.sizes["longitude"]
        return (x, y)


if __name__ == "__main__":
    # # Example usage
    dataset = WeatherBench(var="test", normalize=True, downscaling_factor=2, preload = True)
    print(f"Dataset length: {len(dataset)}")
    print(f"Temporal: {dataset.time_len}")
    target_tensor, input_tensor = dataset.__getitem__(2)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Target tensor shape: {target_tensor.shape}")

    ll, t = dataset.get_grid()
    lat, lon = ll
    print(lat.shape, lon.shape, t.shape)
    x,y = dataset.get_dimensions()
    print(x, y)


    import resource
    point = ""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # you can convert that object to a dictionary
    print(f"{point}: mem (CPU python)={usage[2] / 1024.0}MB;")

    # data_dir = "data"
    # dataset = PDE1D(
    #     data_dir,
    #     pde="Burgers",
    #     var="train",
    #     downscaling_factor=4,
    #     temporal_downscaling_factor=2,
    #     normalize=True,
    #     last_t_steps=2,
    #     select_timesteps="zero",
    #     seed = 0
    # )
    # print(f"Dataset length: {len(dataset)}")
    # print(f"Temporal: {dataset.t_size}")
    # target_tensor, input_tensor = dataset.__getitem__(899)
    # print(f"Input tensor shape: {input_tensor.shape}")
    # print(f"Target tensor shape: {target_tensor.shape}")
    # x_coordinates = dataset.get_coordinates()
    # print(f"x-coordinates shape: {x_coordinates[0].shape}")
    # print(
    #     input_tensor[0].mean(),
    #     target_tensor.mean(),
    #     input_tensor[0].std(),
    #     target_tensor.std(),
    # )
    # x, t = dataset.get_grid()
    # print(x.shape, t.shape)
