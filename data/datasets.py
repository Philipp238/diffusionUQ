import os
from typing import Tuple

import torch
import xarray as xr
from torch.utils.data import Dataset


class PDE1D(Dataset):
    """
    A class used to handle one-dimensional PDE datasets from PDEBench.
    Shape of PDE1D dataset: (samples, time, x)
    """

    def __init__(
        self,
        data_dir: str,
        pde: str,
        test: bool = False,
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
        if test:
            self.filename = "test.nc"
        else:
            self.filename = "train.nc"
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

    def get_coordinates(self) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset["x-coordinate"][:: self.downscaling_factor].values
        return x


if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    dataset = PDE1D(
        data_dir,
        pde="Burgers",
        test=False,
        downscaling_factor=2,
        normalize=True,
        last_t_steps=2,
    )
    print(f"Dataset length: {len(dataset)}")
    target_tensor, input_tensor = dataset.__getitem__(0)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Target tensor shape: {target_tensor.shape}")
    x_coordinates = dataset.get_coordinates()
    print(f"x-coordinates shape: {x_coordinates.shape}")
    print(
        input_tensor[0].mean(),
        target_tensor.mean(),
        input_tensor[0].std(),
        target_tensor.std(),
    )
