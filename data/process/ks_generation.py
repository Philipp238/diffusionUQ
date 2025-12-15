# Description: Script to generate data for the Kuramoto-Sivashinsky equation.

import multiprocessing as mp
import os

# from pde.grids import GridBase
import numpy as np
import pde
import xarray as xr
import yaml
from tqdm import tqdm

PATH = "data/1D_KS_ood"


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
    # state = pde.ScalarField.random_normal(
    #     grid, mean = 2, std = 1
    # )

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


def load_config(path: str = "data/process/ks_config.yaml") -> dict:
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
    if not os.path.exists(f"{PATH}/raw"):
        os.makedirs(f"{PATH}/raw")

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
        np.save(f"{PATH}/raw/ks_data_{i + 1}.npy", result_array)


def aggregate(config: dict, remove: bool, max_steps=104, start_step=6) -> None:
    """Aggregate the raw data into a train and test dataset and save to disk.

    Args:
        config (dict): Configuration dictionary.
    """
    # Check for and create data directory
    if not os.path.exists(f"{PATH}/processed"):
        os.makedirs(f"{PATH}/processed")

    train_split = config["sim"]["train_split"]

    full_data = np.concatenate(
        [np.load(f"{PATH}/raw/{name}") for name in os.listdir(f"{PATH}/raw/")],
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

    ds = ds.isel(t=slice(start_step, max_steps + start_step))

    # Train test split
    n_samples = config["sim"]["num_samples"]
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * train_split)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Create data
    train_data = ds.isel(samples=train_indices)
    test_data = ds.isel(samples=test_indices)

    # Save standardization constants as attributes
    mean = train_data.u.mean().item()
    std = train_data.u.std().item()

    train_data.attrs["mean"] = mean
    train_data.attrs["std"] = std
    test_data.attrs["mean"] = mean
    test_data.attrs["std"] = std

    train_data.to_netcdf(f"{PATH}/processed/train.nc")
    test_data.to_netcdf(f"{PATH}/processed/test.nc")

    if remove:
        os.system(f"rm -r {PATH}/raw")


def main(config, sim=True, remove=False):
    np.random.seed(config["sim"]["seed"])
    # Simulate data
    if sim:
        simulate(config)
    # Aggregate data
    aggregate(config, remove)


if __name__ == "__main__":
    config = load_config()
    main(config, sim=True)
