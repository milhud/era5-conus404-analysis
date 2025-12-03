#!/gpfsm/dnb33/hpmille1/venv/bin/python3

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ks_2samp, linregress
import os
from pathlib import Path
import warnings
import calendar

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
ERA5_FILE = 'era5_2015.nc'
CONUS_FILE = 'conus404_yearly_2010.nc'
OUTPUT_DIR = 'comparison_plots'

# Only processing t2m vs T2
VARIABLE_PAIRS = {
    't2m': 'T2',
    # 'd2m': 'TD2',
    # 'tp': 'RAINNC',
    # 'sp': 'PSFC',
    # 'u10': 'U10',
    # 'v10': 'V10',
}

LAT_MIN, LAT_MAX = 24, 50
LON_MIN, LON_MAX = -125, -66

# Boolean switches
SEPARATE_MONTHLY_STATS = True
SEPARATE_MONTHLY_MAPS = True

def setup_variable_directory(base_output_dir, var_name):
    """Creates a subdirectory for the specific variable (e.g., comparison_plots/t2m)"""
    path = os.path.join(base_output_dir, var_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

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
    """
    Trims dataset to US bounds. 
    - If lat_grid/lon_grid are provided, uses them for masking (Curvilinear/CONUS).
    - Otherwise, looks for standard dims (ERA5).
    """
    # 1. CONUS / Curvilinear Case (Explicit grids provided)
    if lat_grid is not None and lon_grid is not None:
        mask = (lat_grid >= lat_min) & (lat_grid <= lat_max) & \
               (lon_grid >= lon_min) & (lon_grid <= lon_max)
        return data.where(mask, drop=True)
    
    # 2. ERA5 / Rectilinear Case
    if 'latitude' in data.dims and 'longitude' in data.dims:
        return data.sel(
            latitude=slice(lat_max, lat_min), 
            longitude=slice(lon_min, lon_max)
        )
    
    # 3. Fallback
    return data

def load_all_monthly_data(era_ds, conus_ds, era_var, conus_var):
    time_dim = get_time_dimension(conus_ds)
    months = range(1, 13)
    
    era_monthly_data = {}
    conus_monthly_data = {}
    
    for month in months:
        if era_var not in era_ds:
            raise KeyError(f"Variable {era_var} not found in ERA5 dataset")
        if conus_var not in conus_ds:
            raise KeyError(f"Variable {conus_var} not found in CONUS dataset")

        era_month = era_ds[era_var].sel(valid_time=era_ds.valid_time.dt.month == month)
        conus_month = conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month == month})
        
        era_monthly_data[month] = era_month
        conus_monthly_data[month] = conus_month
    
    return era_monthly_data, conus_monthly_data

def compute_global_limits(era_monthly_data, conus_monthly_data):
    all_era_vals = []
    all_conus_vals = []
    
    conus_first_key = list(conus_monthly_data.keys())[0]
    conus_time_dim = get_time_dimension(conus_monthly_data[conus_first_key])

    for month in era_monthly_data.keys():
        era_dims = [d for d in era_monthly_data[month].dims if d in ['valid_time', 'time']]
        era_vals = get_clean_values(era_monthly_data[month].mean(dim=era_dims, skipna=True))
        
        conus_vals = get_clean_values(conus_monthly_data[month].mean(dim=conus_time_dim, skipna=True))
        all_era_vals.extend(era_vals)
        all_conus_vals.extend(conus_vals)
    
    global_min = min(np.min(all_era_vals), np.min(all_conus_vals))
    global_max = max(np.max(all_era_vals), np.max(all_conus_vals))
    
    return global_min, global_max

# --- Plotting Functions ---

def plot_box(ax, era_vals, conus_vals, labels, global_min, global_max, title=None, ylabel=None):
    bp = ax.boxplot([era_vals, conus_vals], labels=labels, 
                    patch_artist=True, showfliers=False, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, labelpad=5)
    
    ax.set_ylim(global_min, global_max)
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=8)
    
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=5)

def plot_ecdf(ax, era_vals, conus_vals, global_min, global_max, title=None):
    era_sorted = np.sort(era_vals)
    conus_sorted = np.sort(conus_vals)
    era_ecdf = np.arange(1, len(era_sorted)+1) / len(era_sorted)
    conus_ecdf = np.arange(1, len(conus_sorted)+1) / len(conus_sorted)
    
    ax.plot(era_sorted, era_ecdf, label="ERA5", color="blue", linewidth=1.5)
    ax.plot(conus_sorted, conus_ecdf, label="C404", color="orange", linewidth=1.5)
    
    ax.set_xlim(global_min, global_max)
    ax.grid(alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
        ax.legend(fontsize=7, loc='lower right')

def plot_qq(ax, x, y, label_x, label_y, global_min, global_max, title=None):
    n = min(len(x), len(y))
    quantiles = np.linspace(0, 1, n)
    x_q = np.quantile(x, quantiles)
    y_q = np.quantile(y, quantiles)
    
    ax.scatter(x_q, y_q, alpha=0.6, s=5, color='darkblue')
    
    slope, intercept, r_value, _, _ = linregress(x_q, y_q)
    fit_line = slope * x_q + intercept
    ax.plot(x_q, fit_line, 'g-', alpha=0.7, linewidth=1.5,
            label=f'R²={r_value**2:.2f}')
    
    ax.set_xlabel(f'{label_x}', fontsize=8)
    ax.tick_params(labelsize=8)
        
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)
    ax.set_aspect('equal', adjustable='box')
    
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=5)

# --- Main Generation Functions ---

def create_map_axis():
    return ccrs.LambertConformal(
        central_longitude=-96.0,
        central_latitude=39.0,
        standard_parallels=(33.0, 45.0)
    )

def add_map_features(ax, lon_min, lon_max, lat_min, lat_max):
    ax.coastlines(resolution='50m', color='black', linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

def generate_monthly_statistics_plots(era_ds, conus_ds, era_var, conus_var, 
                                      save_dir, separate_images):
    print(f"Processing stats: {era_var} vs {conus_var}...")
    
    try:
        era_monthly_data, conus_monthly_data = load_all_monthly_data(era_ds, conus_ds, era_var, conus_var)
    except KeyError as e:
        print(f"  Skipping {era_var}/{conus_var}: {e}")
        return

    global_min, global_max = compute_global_limits(era_monthly_data, conus_monthly_data)
    time_dim = get_time_dimension(conus_ds)
    
    fig, axes = plt.subplots(4, 9, figsize=(28, 16))
    
    for month_idx in range(12):
        month_num = month_idx + 1
        month_name = calendar.month_name[month_num]
        
        row = month_idx // 3
        col_group = month_idx % 3
        
        col_box = col_group * 3
        col_ecdf = col_group * 3 + 1
        col_qq = col_group * 3 + 2
        
        era_dims = [d for d in era_monthly_data[month_num].dims if d in ['valid_time', 'time']]
        era_vals = get_clean_values(era_monthly_data[month_num].mean(dim=era_dims, skipna=True))
        conus_vals = get_clean_values(conus_monthly_data[month_num].mean(dim=time_dim, skipna=True))
        
        # Plotting
        plot_box(axes[row, col_box], era_vals, conus_vals, ['ERA', 'C404'], 
                 global_min, global_max, title=f'{month_name}\nBox Plot', ylabel=f'{era_var}')
        
        plot_ecdf(axes[row, col_ecdf], era_vals, conus_vals, global_min, global_max, 
                  title='ECDF')
        
        plot_qq(axes[row, col_qq], era_vals, conus_vals, 'ERA', 'C404', global_min, global_max, 
                title='Q-Q Plot')

    plt.suptitle(f'{era_var} Monthly Statistics', fontsize=20, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.4, hspace=0.5, left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    output_file = os.path.join(save_dir, f'{era_var}_all_months_stats.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"  Saved: {output_file}")

def generate_monthly_temperature_maps(era_ds, conus_ds, save_dir, separate_images):
    print("Processing temperature maps...")
    
    if 't2m' not in era_ds or 'T2' not in conus_ds:
        print("  Skipping Maps: 't2m' or 'T2' missing.")
        return

    time_dim = get_time_dimension(conus_ds)
    lat_name, lon_name = get_coordinate_names(conus_ds)
    
    # Calculate global limits (TRIMMED) to ensure colorbar is relevant to region
    all_temps = []
    for month in range(1, 13):
        era_dims = [d for d in era_ds['t2m'].dims if d in ['valid_time', 'time']]
        era_month = era_ds['t2m'].sel(valid_time=era_ds.valid_time.dt.month == month).mean(dim=era_dims)
        conus_month = conus_ds['T2'].sel({time_dim: conus_ds[time_dim].dt.month == month}).mean(dim=time_dim)
        
        # Trim temp limits to US region so we don't skew colorbar
        era_month = trim_to_us(era_month, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        conus_month = trim_to_us(conus_month, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, 
                                 lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
        
        all_temps.extend([float(era_month.min()), float(era_month.max()), 
                          float(conus_month.min()), float(conus_month.max())])
    
    vmin = min(all_temps)
    vmax = max(all_temps)
    
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1, 1], wspace=0.1, hspace=0.3, figure=fig)
    
    for month_idx in range(12):
        month_num = month_idx + 1
        month_name = calendar.month_name[month_num]
        
        row = month_idx // 3
        col = month_idx % 3
        
        gs_sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[row, col], 
                                                  width_ratios=[1, 1, 0.05], wspace=0.3)
        
        era_dims = [d for d in era_ds['t2m'].dims if d in ['valid_time', 'time']]
        era_monthly = era_ds['t2m'].sel(valid_time=era_ds.valid_time.dt.month == month_num).mean(dim=era_dims)
        conus_monthly = conus_ds['T2'].sel({time_dim: conus_ds[time_dim].dt.month == month_num}).mean(dim=time_dim)
        
        # --- KEY FIX: Ensure CONUS DataArray has coords attached before trimming ---
        if lat_name in conus_ds and lon_name in conus_ds:
            conus_monthly = conus_monthly.assign_coords({
                lat_name: conus_ds[lat_name],
                lon_name: conus_ds[lon_name]
            })
        
        # Trim Data using explicit grids for CONUS
        era_monthly = trim_to_us(era_monthly, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        conus_monthly = trim_to_us(conus_monthly, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                                   lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
        
        # ERA5 Map
        ax1 = fig.add_subplot(gs_sub[0], projection=create_map_axis())
        p1 = ax1.pcolormesh(era_monthly['longitude'], era_monthly['latitude'], era_monthly,
                       transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
        add_map_features(ax1, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        ax1.set_title(f'{month_name}\nERA5', fontsize=10)
        
        # CONUS Map - FIXED: Pass the trimmed coordinates attached to the object
        ax2 = fig.add_subplot(gs_sub[1], projection=create_map_axis())
        p2 = ax2.pcolormesh(conus_monthly[lon_name], conus_monthly[lat_name], conus_monthly,
                       transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
        add_map_features(ax2, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        ax2.set_title(f'{month_name}\nCONUS404', fontsize=10)
        
        # Colorbar
        cbar_ax = fig.add_subplot(gs_sub[2])
        cbar = fig.colorbar(p2, cax=cbar_ax, extend='both')
        cbar.set_label('K', fontsize=9)
        
    plt.suptitle('Monthly Temperature Comparison', fontsize=20, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    
    output_file = os.path.join(save_dir, 'temperature_all_months.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"  Saved: {output_file}")

def generate_monthly_timeseries(era_ds, conus_ds, era_var, conus_var, save_dir):
    print(f"Processing time series: {era_var} vs {conus_var}...")
    time_dim = get_time_dimension(conus_ds)
    
    try:
        sample_time = pd.to_datetime(era_ds.valid_time.values[0])
        era_year = sample_time.year
    except:
        era_year = "Unknown Year"
    
    fig, axes = plt.subplots(4, 6, figsize=(24, 12))
    
    for month_idx in range(12):
        month_num = month_idx + 1
        month_name = calendar.month_name[month_num]
        
        row = month_idx // 3
        col_group = month_idx % 3
        
        col_era = col_group * 2
        col_conus = col_group * 2 + 1
        
        try:
            # ERA5
            era_month = era_ds[era_var].sel(valid_time=era_ds.valid_time.dt.month == month_num)
            era_times = pd.to_datetime(era_month.valid_time.values)
            reduce_dims = [d for d in era_month.dims if d != 'valid_time']
            era_spatial_mean = era_month.mean(dim=reduce_dims, skipna=True)
            
            axes[row, col_era].plot(era_times, era_spatial_mean, 'b-', linewidth=1.5, label='ERA5')
            axes[row, col_era].set_title(f'{month_name}\nERA5', fontsize=10)
            axes[row, col_era].grid(alpha=0.3)
            axes[row, col_era].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            axes[row, col_era].tick_params(axis='x', rotation=45, labelsize=8)
            
            # CONUS404
            conus_month = conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month == month_num})
            conus_times = pd.to_datetime(conus_month[time_dim].values)
            spatial_dims = [d for d in conus_month.dims if d != time_dim]
            conus_spatial_mean = conus_month.mean(dim=spatial_dims, skipna=True)
            
            axes[row, col_conus].plot(conus_times, conus_spatial_mean, 'r-', linewidth=1.5, label='CONUS')
            axes[row, col_conus].set_title(f'{month_name}\nCONUS', fontsize=10)
            axes[row, col_conus].grid(alpha=0.3)
            axes[row, col_conus].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            axes[row, col_conus].tick_params(axis='x', rotation=45, labelsize=8)

        except KeyError:
            continue

    plt.suptitle(f'{era_var} vs {conus_var} Monthly Time Series ({era_year})', fontsize=20, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.3, hspace=0.6, left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    output_file = os.path.join(save_dir, f'{era_var}_monthly_timeseries.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"  Saved: {output_file}")

def main():
    print("="*40)
    print("ERA5 vs CONUS404 Comparison")
    print("="*40)
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    era_ds, conus_ds = load_datasets(ERA5_FILE, CONUS_FILE)
    
    for era_var, conus_var in VARIABLE_PAIRS.items():
        var_save_dir = setup_variable_directory(OUTPUT_DIR, era_var)
        
        generate_monthly_statistics_plots(era_ds, conus_ds, era_var, conus_var, 
                                          var_save_dir, SEPARATE_MONTHLY_STATS)
        
        generate_monthly_timeseries(era_ds, conus_ds, era_var, conus_var, var_save_dir)
    
    if 't2m' in VARIABLE_PAIRS:
        t2m_save_dir = setup_variable_directory(OUTPUT_DIR, 't2m')
        generate_monthly_temperature_maps(era_ds, conus_ds, t2m_save_dir, SEPARATE_MONTHLY_MAPS)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
