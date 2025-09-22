# Script to download the ERA5 dataset from WeatherBench2. The script filters the data to the Europe grid and the specified date range.

# import apache_beam
import gcsfs
import numpy as np
import pandas as pd
import weatherbench2
import xarray as xr

if __name__ == "__main__":
    # Set parameters
    date_range = pd.date_range(f"2010-01-01", f"2022-12-31T18", freq="6h")
    path = "data/era5/"
    # # Europe grid
    lat_range = np.arange(35, 75, 0.25)
    lon_range = np.append(np.arange(347.5, 360, 0.25), np.arange(0, 42.5, 0.25))

    # ERA 5
    era5_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    variables = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "temperature",
        "geopotential",
        "land_sea_mask",
        "geopotential_at_surface",
    ]
    filename = "era5"

    # Open file from weatherbench and filter
    data = xr.open_zarr(era5_path)
    data_reduced = (
        data[variables]
        .sel(latitude=lat_range, longitude=lon_range, method="nearest")
        .sel(time=date_range, method="nearest")
    )
    data_reduced["geopotential"] = data_reduced["geopotential"].sel(level=500)
    data_reduced["temperature"] = data_reduced["temperature"].sel(level=850)
    # Rechunk and save
    data_reduced.chunk("auto").to_zarr(
        path + filename + ".zarr", zarr_format=2, consolidated=False
    )

    # Save statistics
    statistics = np.zeros((len(variables), 2))
    for i, var in enumerate(variables):
        if var == "land_sea_mask":
            statistics[i, 0] = 0
            statistics[i, 1] = 1
        else:
            statistics[i, 0] = data_reduced[var].mean().compute()
            statistics[i, 1] = data_reduced[var].std().compute()

    np.save(path + "era5_statistics.npy", statistics)

    # Close stream
    data_reduced.close()
    data.close()
