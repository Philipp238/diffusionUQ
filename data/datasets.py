import os
from typing import Tuple

import torch
import xarray as xr
from torch.utils.data import Dataset
import numpy as np


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
    ) -> None:
        """Initialize PDE1D Dataset

        Args:
            data_dir (str): Data directory.
            test (bool, optional): Whether to load train or test data. Defaults to False.
            downscaling_factor (int, optional): Downscaling for spatial resolution. Defaults to 1.
        """
        if pde not in ["Advection", "Burgers", "ReacDiff"]:
            raise ValueError(
                "PDE must be one of the following: 'Advection', 'Burgers', 'ReacDiff'"
            )
        self.pde = pde
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

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        if self.select_timesteps == "zero":
            return len(self.dataset["samples"])
        elif self.select_timesteps == "all":
            pass

    def __getitem__(self, idx: int) -> tuple:
        """Returns the idx-th element of the dataset

        Args:
            idx (int): Index of the element to be returned

        Returns:
            tuple: Tuple containing the input and output tensors
        """
        input_downscaled = self.dataset.u[
            :, :: self.temporal_downscaling_factor, :: self.downscaling_factor
        ]
        target_downscaled = self.dataset.u[
            :, :: self.temporal_downscaling_factor, :: self.downscaling_factor
        ]

        if self.select_timesteps == "zero":
            input = input_downscaled[idx, 0 : self.last_t_steps].to_numpy()
            target = target_downscaled[idx, self.last_t_steps].to_numpy()

            # Normalize
            if self.normalize:
                input = (input - self.mean) / self.std
                target = (target - self.mean) / self.std

            # Get residual
            target = target - input[-1]

        # Add grid
        grid = self.dataset["x-coordinate"][:: self.downscaling_factor].to_numpy()

        input_tensor = torch.cat(
            [torch.tensor(input), torch.tensor(grid).unsqueeze(0)], dim=0
        ).float()
        target_tensor = torch.tensor(target).unsqueeze(0)

        return target_tensor, input_tensor
    
    def get_trajectory(self, idx:int, length:int = 10) -> torch.Tensor:
        """Returns a trajectory of the idx-th element of the dataset

        Args:
            idx (int): Index of the element to be returned
            length (int, optional): Length of the trajectory. Defaults to 10.

        Returns:
            torch.Tensor: Trajectory tensor
        """
        input_downscaled = self.dataset.u[
            :, :: self.temporal_downscaling_factor, :: self.downscaling_factor
        ]
        target_downscaled = self.dataset.u[
            :, :: self.temporal_downscaling_factor, :: self.downscaling_factor
        ]
        input = input_downscaled[idx, 0 : self.last_t_steps].to_numpy()
        target = target_downscaled[idx, self.last_t_steps:(self.last_t_steps + length)].to_numpy()

        # Normalize
        if self.normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std


        # Add grid
        grid = self.dataset["x-coordinate"][:: self.downscaling_factor].to_numpy()

        input_tensor = torch.cat(
            [torch.tensor(input), torch.tensor(grid).unsqueeze(0)], dim=0
        ).float()
        target_tensor = torch.tensor(target).unsqueeze(0)

        return target_tensor, input_tensor


    def get_coordinates(self) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset["x-coordinate"][:: self.downscaling_factor].values
        return (x,)
    
    def get_dimensions(self) -> Tuple:
        x = len(self.dataset["x-coordinate"][:: self.downscaling_factor].values)
        return (x,)
    
    def destandardize_output(self, u):
        if hasattr(self, "mean") and hasattr(self, "std"):
            return u * self.std + self.mean
        else:
            return u
    

class PDE2D(Dataset):
    """
    A class used to handle two-dimensional PDE datasets from PDEBench.
    Shape of PDE1D dataset: (samples, channel, x, y)
    """

    def __init__(
        self,
        data_dir: str,
        pde: str,
        var: str = "train",
        downscaling_factor: int = 1,
        normalize=True,
    ) -> None:
        """Initialize PDE1D Dataset

        Args:
            data_dir (str): Data directory.
            test (bool, optional): Whether to load train or test data. Defaults to False.
            downscaling_factor (int, optional): Downscaling for spatial resolution. Defaults to 1.
        """
        if pde not in ["DarcyFlow"]:
            raise ValueError(
                "PDE must be one of the following: DarcyFlow"
            )
        self.pde = pde
        self.filename = f"{var}.nc"
        self.path = os.path.join(data_dir, f"2D_{self.pde}/processed/")
        self.dataset = xr.open_dataset(self.path + self.filename)
        assert isinstance(downscaling_factor, int), "Scaling factor must be Integer"
        self.downscaling_factor = downscaling_factor
        self.normalize = normalize
        self.data_dir = data_dir

        # Get normalization
        self.mean = self.dataset.attrs.get("mean", None)
        self.std = self.dataset.attrs.get("std", None)

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.dataset["samples"])


    def __getitem__(self, idx: int) -> tuple:
        """Returns the idx-th element of the dataset

        Args:
            idx (int): Index of the element to be returned

        Returns:
            tuple: Tuple containing the input and output tensors
        """
        input_downscaled = self.dataset.a[
            :, :: self.downscaling_factor, :: self.downscaling_factor
        ]
        target_downscaled = self.dataset.u[
            :,:, :: self.downscaling_factor, :: self.downscaling_factor
        ]


        input = input_downscaled[idx].to_numpy()
        target = target_downscaled[idx].to_numpy()

        # Normalize
        if self.normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std


        # Add grid
        x = self.dataset["x-coordinate"][:: self.downscaling_factor].to_numpy()
        y = self.dataset["y-coordinate"][:: self.downscaling_factor].to_numpy()
        grid = np.stack(np.meshgrid(x, y))
        

        input_tensor = torch.cat(
            [torch.tensor(input).unsqueeze(0), torch.tensor(grid)], dim=0
        ).float()
        target_tensor = torch.tensor(target)
        return target_tensor, input_tensor

    def get_coordinates(self) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset["x-coordinate"][:: self.downscaling_factor].values
        y = self.dataset["y-coordinate"][:: self.downscaling_factor].values
        return (x,y)
    
    def get_dimensions(self) -> Tuple:
        x = len(self.dataset["x-coordinate"][:: self.downscaling_factor].values)
        y = len(self.dataset["y-coordinate"][:: self.downscaling_factor].values)
        return (x,y)

    def destandardize_output(self, u):
        if hasattr(self, "mean") and hasattr(self, "std"):
            return u * self.std + self.mean
        else:
            return u


if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    dataset = PDE1D(
        data_dir,
        pde="Advection",
        var="val",
        downscaling_factor=2,
        normalize=True,
        last_t_steps=2,
    )
    dataset = PDE2D(
        data_dir,
        pde="DarcyFlow",
        var="val",
        downscaling_factor=1,
    )
    print(f"Dataset length: {len(dataset)}")
    target_tensor, input_tensor = dataset.__getitem__(0)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Target tensor shape: {target_tensor.shape}")
    x_coordinates = dataset.get_coordinates()
    print(f"x-coordinates shape: {x_coordinates[0].shape}")
    print(
        input_tensor[0].mean(),
        target_tensor.mean(),
        input_tensor[0].std(),
        target_tensor.std(),
    )
