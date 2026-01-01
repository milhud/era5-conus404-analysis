from setup import get_time_dimension,get_coordinate_names,trim_to_us,get_clean_values,compute_global_limits
from plotting import create_map_axis,add_map_features,plot_box,plot_ecdf,plot_qq

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import os
from pathlib import Path

# ERA5 (key) : CONUS404 (value)
VARIABLE_PAIRS = {
    't2m': 'T2',
    'd2m': 'TD2',
    'sp': 'PSFC',
    'u10': 'U10',
    'v10': 'V10',
    'lai': 'LAI',
    'tp': 'ACRAINLSM',
}

# Units for labeling
VARIABLE_UNITS = {
    't2m': 'K',
    'd2m': 'K',
    'sp': 'Pa',
    'u10': 'm/s',
    'v10': 'm/s',
    'lai': 'Index',
    'tp': 'mm',
    'z': 'm²/s²'
}

LAT_MIN, LAT_MAX = 24, 50
LON_MIN, LON_MAX = -125, -66

def setup_yearly_directories(base_output_dir):
    """Creates directory structure for Yearly data."""
    yearly_base = os.path.join(base_output_dir, 'yearly')
    Path(yearly_base).mkdir(parents=True, exist_ok=True)
    return yearly_base

def generate_yearly_single_variable(era_ds, conus_ds, era_var, conus_var, yearly_base_dir):
    """Generates Stats, Maps, and Timeseries for the FULL YEAR for a single variable."""
    print(f"Processing YEARLY data for {era_var}...")
    
    var_dir = os.path.join(yearly_base_dir, era_var)
    Path(var_dir).mkdir(exist_ok=True)
    time_dim = get_time_dimension(conus_ds)
    lat_name, lon_name = get_coordinate_names(conus_ds)
    unit = VARIABLE_UNITS.get(era_var, '')
    
    # 1. Yearly Statistics
    era_dims = [d for d in era_ds[era_var].dims if d in ['valid_time', 'time']]
    conus_time_dim = get_time_dimension(conus_ds)
    
    era_agg = era_ds[era_var].mean(dim=era_dims, skipna=True)
    conus_agg = conus_ds[conus_var].mean(dim=conus_time_dim, skipna=True)
    
    era_vals = get_clean_values(era_agg)
    conus_vals = get_clean_values(conus_agg)
    gmin, gmax = min(era_vals.min(), conus_vals.min()), max(era_vals.max(), conus_vals.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_box(axes[0], era_vals, conus_vals, ['ERA', 'C404'], gmin, gmax, title='Yearly Box Plot', ylabel=f'{era_var} ({unit})')
    plot_ecdf(axes[1], era_vals, conus_vals, gmin, gmax, title='Yearly ECDF', unit_label=unit)
    plot_qq(axes[2], era_vals, conus_vals, 'ERA', 'C404', gmin, gmax, title='Yearly Q-Q Plot', unit_label=unit)
    plt.savefig(os.path.join(var_dir, f'yearly_stats_{era_var}.png'), dpi=300)
    plt.close()

    # 2. Yearly Maps
    era_map = trim_to_us(era_agg, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    conus_map = trim_to_us(conus_agg, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
    
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
    
    ax1 = fig.add_subplot(gs[0], projection=create_map_axis())
    ax1.pcolormesh(era_map['longitude'], era_map['latitude'], era_map, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=gmin, vmax=gmax)
    add_map_features(ax1, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    ax1.set_title("ERA5 Yearly Mean")
    
    ax2 = fig.add_subplot(gs[1], projection=create_map_axis())
    p2 = ax2.pcolormesh(conus_map[lon_name], conus_map[lat_name], conus_map, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=gmin, vmax=gmax)
    add_map_features(ax2, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    ax2.set_title("CONUS404 Yearly Mean")
    
    cbar = fig.colorbar(p2, cax=fig.add_subplot(gs[2]), extend='both')
    cbar.set_label(unit)
    plt.savefig(os.path.join(var_dir, f'yearly_map_{era_var}.png'), dpi=300)
    plt.close()