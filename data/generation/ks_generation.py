# Description: Script to generate data for the Kuramoto-Sivashinsky equation.

import pde

# from pde.grids import GridBase
import numpy as np
import yaml
import multiprocessing as mp
import os
from tqdm import tqdm
import xarray as xr


def get_sim(grid, n_steps: int, t_range: int = 150, dt: float = 0.001) -> np.ndarray:
    """Simulate an instance of the Kuramoto-Sivashinsky equation.
    Args:
        grid (GridBase): The grid over which to simulate the PDE.
        n_steps (int): The timesteps to track.
        t_range (int, optional): The time range up to which to simulate. Defaults to 150.
        dt (float, optional): The temporal resolution of the solver. Defaults to 0.001.

    Returns:
        np.ndarray: Simulated data.
    """
    eq = pde.PDE({"u": "-u * d_dx(u) / 2 - laplace(u + laplace(u))"})
    state = pde.ScalarField.random_uniform(
        grid, vmin=-1, vmax=1
    )  # generate initial condition
    # solve the system
    storage = pde.MemoryStorage()
    result = eq.solve(
        state,
        t_range=t_range,
        dt=dt,
        adaptive=True,
        tracker=[storage.tracker(n_steps)],
    )
    res = np.array(storage.data)
    return res


def load_config(path: str = "data/generation/ks_config.yaml") -> dict:
    """Load the configuration file for the simulation.

    Args:
        path (str, optional): Path to config file. Defaults to "data/generation/ks_config.yaml".

    Returns:
        dict: Configuration dictionary.
    """
    with open(path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


def simulate(config: dict) -> None:
    """Simulate the Kuramoto-Sivashinsky equation and save the raw data to disk.

    Args:
        config (dict): Configuration dictionary.
    """
    # Check for number of cores
    assert config["sim"]["n_cores"] <= mp.cpu_count()
    n_cores = config["sim"]["n_cores"]
    # Check for and create data directory
    if not os.path.exists("data/1D_KS/raw"):
        os.makedirs("data/1D_KS/raw")

    # Create grid
    grid = pde.CartesianGrid(
        [(0, config["pde"]["L"])], [config["pde"]["res"]], periodic=True
    )
    # Number of subiterations
    n_sub = int(config["sim"]["num_samples"] / config["sim"]["subfile_size"])
    # Number of timesteps to record
    n_steps = np.arange(0, config["pde"]["t_max"], config["pde"]["dt"])

    # Iterate over subiterations
    for i in tqdm(range(0, n_sub)):
        pool = mp.Pool(n_cores)
        results = []
        result_objects = [
            pool.apply_async(get_sim, args=(grid, n_steps, config["pde"]["t_max"]))
            for i in range(config["sim"]["subfile_size"])
        ]
        results = [r.get() for r in result_objects]
        pool.close()
        pool.join()
        result_array = np.array(results)
        np.save(f"data/1D_KS/raw/ks_data_{i + 1}.npy", result_array)


def aggregate(config: dict, remove: bool) -> None:
    """Aggregate the raw data into a train and test dataset and save to disk.

    Args:
        config (dict): Configuration dictionary.
    """
    # Check for and create data directory
    if not os.path.exists("data/1D_KS/processed"):
        os.makedirs("data/1D_KS/processed")

    train_split = config["sim"]["train_split"]

    full_data = np.concatenate(
        [np.load(f"data/1D_KS/raw/{name}") for name in os.listdir("data/1D_KS/raw/")],
        axis=0,
    )
    t = np.arange(0, config["pde"]["t_max"], config["pde"]["dt"])
    x = np.linspace(0, config["pde"]["L"], config["pde"]["res"])
    data_array = xr.DataArray(full_data, dims=["samples", "t", "x"])
    t_coordinate = xr.DataArray(t, dims=["t"])
    x_coordinate = xr.DataArray(x, dims=["x"])
    ds = xr.Dataset(
        {"u": data_array, "x-coordinate": x_coordinate, "t-coordinate": t_coordinate}
    )

    # Train test split
    n_samples = config["sim"]["num_samples"]
    indices = np.random.permutation(n_samples)
    n_train_val = int(n_samples * train_split)
    n_train = int(n_train_val * train_split)
    n_val = n_train_val - n_train
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    # Create data
    train_data = ds.isel(samples=train_indices)
    val_data = ds.isel(samples=val_indices)
    test_data = ds.isel(samples=test_indices)

    # Save standardization constants as attributes
    mean = train_data.u.mean().item()
    std = train_data.u.std().item()

    train_data.attrs["mean"] = mean
    train_data.attrs["std"] = std
    val_data.attrs["mean"] = mean
    val_data.attrs["std"] = std
    test_data.attrs["mean"] = mean
    test_data.attrs["std"] = std

    train_data.to_netcdf("data/1D_KS/processed/train.nc")
    val_data.to_netcdf("data/1D_KS/processed/val.nc")
    test_data.to_netcdf("data/1D_KS/processed/test.nc")

    if remove:
        os.system(f"rm -r data/1D_KS/raw")


def main(config, sim=True, remove=False):
    np.random.seed(config["sim"]["seed"])
    # Simulate data
    if sim:
        simulate(config)
    # Aggregate data
    aggregate(config, remove)


if __name__ == "__main__":
    config = load_config()
    main(config, sim=False)
