import os
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

def load_datasets(era_file, conus_file):
    try:
        era_ds = xr.open_dataset(era_file)
        conus_ds = xr.open_dataset(conus_file)
        return era_ds, conus_ds
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        exit(1)

def get_time_dimension(ds):
    return 'Time' if 'Time' in ds.dims else 'time'

def get_coordinate_names(ds):
    lat_name = 'XLAT' if 'XLAT' in ds else 'lat'
    lon_name = 'XLONG' if 'XLONG' in ds else 'lon'
    return lat_name, lon_name

def get_clean_values(data):
    vals = data.values.flatten()
    return vals[np.isfinite(vals)]

def trim_to_us(data, lat_min, lat_max, lon_min, lon_max, lat_grid=None, lon_grid=None):
    if lat_grid is not None and lon_grid is not None:
        mask = (
            (lat_grid >= lat_min) & (lat_grid <= lat_max) &
            (lon_grid >= lon_min) & (lon_grid <= lon_max)
        )
        return data.where(mask, drop=True)
    
    if 'latitude' in data.dims and 'longitude' in data.dims:
        return data.sel(
            latitude=slice(lat_max, lat_min), 
            longitude=slice(lon_min, lon_max)
        )
    return data

def convert_units(data, var_name):
    # convert era5 tp from meters to millimeters
    if var_name == 'tp':
        return data * 1000.0
    # convert surface pressure from pascals to hectopascals
    elif var_name == 'sp':
        return data / 100.0
    else:
        return data

def setup_directories(base_output_dir, var_name, separate_mode):
    base_var_dir = os.path.join(base_output_dir, var_name)
    Path(base_var_dir).mkdir(parents=True, exist_ok=True)
    
    paths = {
        'base': base_var_dir,
        'stats': base_var_dir,
        'maps': base_var_dir,
        'timeseries': base_var_dir
    }
    
    if separate_mode:
        for subtype in ['stats', 'maps', 'timeseries']:
            sub_path = os.path.join(base_var_dir, subtype)
            Path(sub_path).mkdir(exist_ok=True)
            paths[subtype] = sub_path
            
    return paths

def compute_global_limits(era_monthly_data, conus_monthly_data, era_var):
    """Computes global min/max with unit conversion for TP."""
    all_era_vals = []
    all_conus_vals = []
    
    conus_first_key = list(conus_monthly_data.keys())[0]
    conus_time_dim = get_time_dimension(conus_monthly_data[conus_first_key])

    for month in era_monthly_data.keys():
        era_dims = [d for d in era_monthly_data[month].dims if d in ['valid_time', 'time']]
        
        if era_var == 'tp':
            # Sum across time, then multiply by 1000 for Meters -> Millimeters
            era_agg = era_monthly_data[month].sum(dim=era_dims, skipna=True) * 1000
            conus_agg = conus_monthly_data[month].sum(dim=conus_time_dim, skipna=True)
        else:
            era_agg = era_monthly_data[month].mean(dim=era_dims, skipna=True)
            conus_agg = conus_monthly_data[month].mean(dim=conus_time_dim, skipna=True)

        era_vals = get_clean_values(era_agg)
        conus_vals = get_clean_values(conus_agg)
        
        all_era_vals.extend(era_vals)
        all_conus_vals.extend(conus_vals)
    
    if not all_era_vals or not all_conus_vals:
        return 0, 1

    global_min = min(np.min(all_era_vals), np.min(all_conus_vals))
    global_max = max(np.max(all_era_vals), np.max(all_conus_vals))
    
    return global_min, global_max