#!/usr/local/other/GEOSpyD/24.3.0-0/2024-08-29/envs/py3.12/bin/python3

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
ERA5_FILE = 'era5_2010.nc'
CONUS_FILE = 'conus404_yearly_2010.nc'
OUTPUT_DIR = 'comparison_plots'

# ERA5 (key) : CONUS404 (value)
VARIABLE_PAIRS = {
    't2m': 'T2'#,          # 2m Temperature
    #'d2m': 'TD2',         # 2m Dewpoint Temperature
    #'sp': 'PSFC',         # Surface Pressure
    #'u10': 'U10',         # 10m U-Wind Component
    #'v10': 'V10',         # 10m V-Wind Component
    #'lai': 'LAI',         # Leaf Area Index
    #'tp': 'ACRAINLSM',    # Total Precipitation
    # 'z': 'Z',             # Geopotential (Commented out)
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

# Boolean switch (Controls Monthly Stats, Maps, and Time Series separation)
SEPARATE_IMAGES = True 

def load_all_seasonal_data(era_ds, conus_ds, era_var, conus_var):
    """Load and prepare seasonal data for all four seasons."""
    time_dim = get_time_dimension(conus_ds)
    lat_name, lon_name = get_coordinate_names(conus_ds)
    
    seasons = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11]
    }
    
    era_seasonal_data = {}
    conus_seasonal_data = {}
    
    for season_name, months in seasons.items():
        # ERA5 seasonal mean
        era_season = era_ds[era_var].sel(valid_time=era_ds.valid_time.dt.month.isin(months)).mean(dim='valid_time')
        era_season = trim_to_us(era_season, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        era_seasonal_data[season_name] = era_season
        
        # CONUS seasonal mean
        conus_season = conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month.isin(months)}).mean(dim=time_dim)
        if lat_name in conus_ds and lon_name in conus_ds:
            conus_season = conus_season.assign_coords({lat_name: conus_ds[lat_name], lon_name: conus_ds[lon_name]})
        conus_season = trim_to_us(
            conus_season, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
            lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name]
        )
        conus_seasonal_data[season_name] = conus_season
    
    return era_seasonal_data, conus_seasonal_data

def setup_directories(base_output_dir, var_name, separate_mode):
    """Creates directory structure for Monthly data."""
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

def setup_yearly_directories(base_output_dir):
    """Creates directory structure for Yearly data."""
    yearly_base = os.path.join(base_output_dir, 'yearly')
    Path(yearly_base).mkdir(parents=True, exist_ok=True)
    return yearly_base

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
    """Trims dataset to US bounds."""
    if lat_grid is not None and lon_grid is not None:
        mask = (
            (lat_grid >= lat_min) & (lat_grid <= lat_max) &
            (lon_grid >= lon_min) & (lon_grid <= lon_max)
        )
        return data.where(mask, drop=False)
    
    if 'latitude' in data.dims and 'longitude' in data.dims:
        return data.sel(
            latitude=slice(lat_max, lat_min), 
            longitude=slice(lon_min, lon_max)
        )
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

def load_all_seasonal_data(era_ds, conus_ds, era_var, conus_var):
    time_dim = get_time_dimension(conus_ds)
    seasons = {"winter":[1,2,12],"spring":{3,4,5},"summer":{6,7,8},"autumn":{9,10,11}}

    era_seasonal_data = {}
    conus_seasonal_data = {}

    for season, months in seasons:
        if era_var not in era_ds:
            raise KeyError(f"Variable {era_var} not found in ERA5 dataset")
        if conus_var not in conus_ds:
            raise KeyError(f"Variable {conus_var} not found in CONUS dataset")
        
        era_season = era_ds[era_var].sel(valid_time = era_ds.valid_time.dt.month.isin(months))
        conus_season = conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month.isin(months)})

        era_seasonal_data[season] = era_season
        conus_seasonal_data[season] = conus_season

    return era_seasonal_data, conus_seasonal_data    

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

# --- Plotting Functions ---

def plot_box(ax, era_vals, conus_vals, labels, global_min, global_max, title=None, ylabel=None):
    bp = ax.boxplot([era_vals, conus_vals], labels=labels, 
                    patch_artist=True, showfliers=False, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    for median in bp['medians']:
        median.set_color('darkred')
        median.set_linewidth(2)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    
    if global_min is not None and global_max is not None:
        y_range = global_max - global_min
        pad = y_range * 0.05
        ax.set_ylim(global_min - pad, global_max + pad)

    ax.grid(alpha=0.3, axis='y')
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

def plot_ecdf(ax, era_vals, conus_vals, global_min, global_max, title=None, unit_label=''):
    era_sorted = np.sort(era_vals)
    conus_sorted = np.sort(conus_vals)
    era_ecdf = np.arange(1, len(era_sorted)+1) / len(era_sorted)
    conus_ecdf = np.arange(1, len(conus_sorted)+1) / len(conus_sorted)
    
    ax.plot(era_sorted, era_ecdf, label="ERA5", color="blue", linewidth=1.5)
    ax.plot(conus_sorted, conus_ecdf, label="C404", color="orange", linewidth=1.5)
    
    if global_min is not None and global_max is not None:
        ax.set_xlim(global_min, global_max)
    ax.grid(alpha=0.3)
    ax.set_xlabel(f'Value ({unit_label})', fontsize=9)
    ax.set_ylabel('Probability', fontsize=9)
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.legend(fontsize=8, loc='lower right')

def plot_qq(ax, x, y, label_x, label_y, global_min, global_max, title=None, unit_label=''):
    n = min(len(x), len(y))
    quantiles = np.linspace(0, 1, n)
    x_q = np.quantile(x, quantiles)
    y_q = np.quantile(y, quantiles)
    
    ax.scatter(x_q, y_q, alpha=0.6, s=5, color='darkblue')
    
    if len(x_q) > 1 and (x_q.std() > 0 or y_q.std() > 0):
        slope, intercept, r_value, _, _ = linregress(x_q, y_q)
        fit_line = slope * x_q + intercept
        ax.plot(x_q, fit_line, 'g-', alpha=0.7, linewidth=1.5,
                label=f'R²={r_value**2:.2f}')
    
    ax.set_xlabel(f'{label_x} ({unit_label})', fontsize=9)
    ax.set_ylabel(f'{label_y} ({unit_label})', fontsize=9)
        
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    if global_min is not None and global_max is not None:
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
    ax.set_aspect('equal', adjustable='box')
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

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

# --- MONTHLY GENERATION FUNCTIONS ---

def generate_monthly_temperature_maps(era_ds, conus_ds, dirs, separate_images):
    """Generate temperature maps."""
    print("Processing monthly temperature maps...")
    if 't2m' not in era_ds or 'T2' not in conus_ds: return
    time_dim, (lat_name, lon_name) = get_time_dimension(conus_ds), get_coordinate_names(conus_ds)
    unit = VARIABLE_UNITS.get('t2m', 'K')
    
    all_temps = []
    for month in range(1, 13):
        era_dims = [d for d in era_ds['t2m'].dims if d in ['valid_time', 'time']]
        era_month = trim_to_us(era_ds['t2m'].sel(valid_time=era_ds.valid_time.dt.month == month).mean(dim=era_dims), LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        conus_month = trim_to_us(conus_ds['T2'].sel({time_dim: conus_ds[time_dim].dt.month == month}).mean(dim=time_dim), LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
        all_temps.extend([float(era_month.min()), float(era_month.max()), float(conus_month.min()), float(conus_month.max())])
    vmin, vmax = min(all_temps), max(all_temps)

    if separate_images:
        for month_idx in range(12):
            month_num = month_idx + 1
            month_name = calendar.month_name[month_num]
            fig = plt.figure(figsize=(15, 6))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
            
            era_dims = [d for d in era_ds['t2m'].dims if d in ['valid_time', 'time']]
            era_m = trim_to_us(era_ds['t2m'].sel(valid_time=era_ds.valid_time.dt.month == month_num).mean(dim=era_dims), LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            conus_m = conus_ds['T2'].sel({time_dim: conus_ds[time_dim].dt.month == month_num}).mean(dim=time_dim)
            if lat_name in conus_ds and lon_name in conus_ds: conus_m = conus_m.assign_coords({lat_name: conus_ds[lat_name], lon_name: conus_ds[lon_name]})
            conus_m = trim_to_us(conus_m, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
            
            ax1 = fig.add_subplot(gs[0], projection=create_map_axis())
            ax1.pcolormesh(era_m['longitude'], era_m['latitude'], era_m, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
            add_map_features(ax1, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
            ax1.set_title('ERA5', fontsize=12)
            
            ax2 = fig.add_subplot(gs[1], projection=create_map_axis())
            p2 = ax2.pcolormesh(conus_m[lon_name], conus_m[lat_name], conus_m, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
            add_map_features(ax2, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
            ax2.set_title('CONUS404', fontsize=12)
            
            cbar = fig.colorbar(p2, cax=fig.add_subplot(gs[2]), extend='both')
            cbar.set_label(f'Temperature ({unit})', fontsize=10)
            plt.suptitle(f'{month_name} Temperature Comparison', fontsize=16, fontweight='bold', y=0.95)
            plt.savefig(os.path.join(dirs['maps'], f'map_t2m_month{month_num:02d}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    else:
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1, 1], wspace=0.1, hspace=0.3, figure=fig)
        for month_idx in range(12):
            month_num = month_idx + 1
            month_name = calendar.month_name[month_num]
            row, col = month_idx // 3, month_idx % 3
            gs_sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[row, col], width_ratios=[1, 1, 0.05], wspace=0.3)
            
            era_dims = [d for d in era_ds['t2m'].dims if d in ['valid_time', 'time']]
            era_m = trim_to_us(era_ds['t2m'].sel(valid_time=era_ds.valid_time.dt.month == month_num).mean(dim=era_dims), LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            conus_m = conus_ds['T2'].sel({time_dim: conus_ds[time_dim].dt.month == month_num}).mean(dim=time_dim)
            if lat_name in conus_ds and lon_name in conus_ds: conus_m = conus_m.assign_coords({lat_name: conus_ds[lat_name], lon_name: conus_ds[lon_name]})
            conus_m = trim_to_us(conus_m, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
            
            ax1 = fig.add_subplot(gs_sub[0], projection=create_map_axis())
            ax1.pcolormesh(era_m['longitude'], era_m['latitude'], era_m, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
            add_map_features(ax1, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
            ax1.set_title(f'{month_name}\nERA5', fontsize=10)
            
            ax2 = fig.add_subplot(gs_sub[1], projection=create_map_axis())
            p2 = ax2.pcolormesh(conus_m[lon_name], conus_m[lat_name], conus_m, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
            add_map_features(ax2, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
            ax2.set_title(f'{month_name}\nCONUS404', fontsize=10)
            
            fig.colorbar(p2, cax=fig.add_subplot(gs_sub[2]), extend='both').set_label(f'K', fontsize=9)
        plt.suptitle('Monthly Temperature Comparison', fontsize=20, fontweight='bold', y=0.98)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
        plt.savefig(os.path.join(dirs['base'], 'map_t2m_all_months.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_monthly_statistics_plots(era_ds, conus_ds, era_var, conus_var, 
                                      dirs, separate_images):
    print(f"Processing monthly stats: {era_var} vs {conus_var}...")
    
    try:
        era_monthly_data, conus_monthly_data = load_all_monthly_data(era_ds, conus_ds, era_var, conus_var)
    except KeyError as e:
        print(f"  Skipping {era_var}/{conus_var}: {e}")
        return

    global_min, global_max = compute_global_limits(era_monthly_data, conus_monthly_data, era_var)
    time_dim = get_time_dimension(conus_ds)
    unit = VARIABLE_UNITS.get(era_var, '')
    
    if separate_images:
        for month_idx in range(12):
            month_num = month_idx + 1
            month_name = calendar.month_name[month_num]
            
            era_dims = [d for d in era_monthly_data[month_num].dims if d in ['valid_time', 'time']]
            
            if era_var == 'tp':
                era_agg = era_monthly_data[month_num].sum(dim=era_dims, skipna=True) * 1000
                conus_agg = conus_monthly_data[month_num].sum(dim=time_dim, skipna=True)
            else:
                era_agg = era_monthly_data[month_num].mean(dim=era_dims, skipna=True)
                conus_agg = conus_monthly_data[month_num].mean(dim=time_dim, skipna=True)

            era_vals = get_clean_values(era_agg)
            conus_vals = get_clean_values(conus_agg)
            
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            plot_box(axes[0], era_vals, conus_vals, ['ERA', 'C404'], global_min, global_max, title='Box Plot', ylabel=f'{era_var} ({unit})')
            plot_ecdf(axes[1], era_vals, conus_vals, global_min, global_max, title='ECDF', unit_label=unit)
            plot_qq(axes[2], era_vals, conus_vals, 'ERA', 'C404', global_min, global_max, title='Q-Q Plot', unit_label=unit)
            plt.suptitle(f'{month_name} Statistics - {era_var}', fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            output_file = os.path.join(dirs['stats'], f'stats_{era_var}_month{month_num:02d}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
    else:
        fig, axes = plt.subplots(4, 9, figsize=(28, 16))
        for month_idx in range(12):
            month_num = month_idx + 1
            month_name = calendar.month_name[month_num]
            row = month_idx // 3
            col_group = month_idx % 3
            era_dims = [d for d in era_monthly_data[month_num].dims if d in ['valid_time', 'time']]
            
            if era_var == 'tp':
                era_agg = era_monthly_data[month_num].sum(dim=era_dims, skipna=True) * 1000
                conus_agg = conus_monthly_data[month_num].sum(dim=time_dim, skipna=True)
            else:
                era_agg = era_monthly_data[month_num].mean(dim=era_dims, skipna=True)
                conus_agg = conus_monthly_data[month_num].mean(dim=time_dim, skipna=True)

            era_vals = get_clean_values(era_agg)
            conus_vals = get_clean_values(conus_agg)

            plot_box(axes[row, col_group*3], era_vals, conus_vals, ['ERA', 'C404'], global_min, global_max, title=f'{month_name}\nBox Plot', ylabel=f'{era_var} ({unit})')
            plot_ecdf(axes[row, col_group*3+1], era_vals, conus_vals, global_min, global_max, title='ECDF', unit_label=unit)
            plot_qq(axes[row, col_group*3+2], era_vals, conus_vals, 'ERA', 'C404', global_min, global_max, title='Q-Q Plot', unit_label=unit)
        plt.suptitle(f'{era_var} Monthly Statistics', fontsize=20, fontweight='bold', y=0.98)
        plt.subplots_adjust(wspace=0.4, hspace=0.5, left=0.05, right=0.95, top=0.92, bottom=0.05)
        output_file = os.path.join(dirs['base'], f'stats_{era_var}_all_months.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

def generate_monthly_timeseries(era_ds, conus_ds, era_var, conus_var, dirs, separate_images):
    print(f"Processing monthly time series: {era_var} vs {conus_var}...")
    time_dim = get_time_dimension(conus_ds)
    unit = VARIABLE_UNITS.get(era_var, '')
    try: era_year = pd.to_datetime(era_ds.valid_time.values[0]).year
    except: era_year = "Unknown Year"

    if separate_images:
        for month_idx in range(12):
            month_num = month_idx + 1
            try:
                era_month = era_ds[era_var].sel(valid_time=era_ds.valid_time.dt.month == month_num)
                conus_month = conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month == month_num})
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                era_times = pd.to_datetime(era_month.valid_time.values)
                reduce_dims = [d for d in era_month.dims if d != 'valid_time']
                
                # UNIT CONVERSION FOR TIMESERIES
                era_ts_data = era_month.mean(dim=reduce_dims, skipna=True)
                if era_var == 'tp':
                    era_ts_data = era_ts_data * 1000
                
                axes[0].plot(era_times, era_ts_data, 'b-', linewidth=1.5)
                axes[0].set_title('ERA5', fontsize=12); axes[0].grid(alpha=0.3)
                axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                axes[0].set_ylabel(f'{era_var} ({unit})')
                
                conus_times = pd.to_datetime(conus_month[time_dim].values)
                spatial_dims = [d for d in conus_month.dims if d != time_dim]
                axes[1].plot(conus_times, conus_month.mean(dim=spatial_dims, skipna=True), 'r-', linewidth=1.5)
                axes[1].set_title('CONUS', fontsize=12); axes[1].grid(alpha=0.3)
                axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                axes[1].set_ylabel(f'{era_var} ({unit})')
                
                plt.suptitle(f'{calendar.month_name[month_num]} Time Series - {era_var}', fontsize=16, fontweight='bold', y=0.98)
                plt.savefig(os.path.join(dirs['timeseries'], f'timeseries_{era_var}_month{month_num:02d}.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except KeyError: continue
    else:
        fig, axes = plt.subplots(4, 6, figsize=(24, 12))
        for month_idx in range(12):
            month_num = month_idx + 1
            row, col_group = month_idx // 3, month_idx % 3
            try:
                era_month = era_ds[era_var].sel(valid_time=era_ds.valid_time.dt.month == month_num)
                conus_month = conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month == month_num})
                
                era_times = pd.to_datetime(era_month.valid_time.values)
                
                # UNIT CONVERSION FOR TIMESERIES (COMPACT)
                era_ts_data = era_month.mean(dim=[d for d in era_month.dims if d != 'valid_time'], skipna=True)
                if era_var == 'tp': era_ts_data = era_ts_data * 1000

                axes[row, col_group*2].plot(era_times, era_ts_data, 'b-')
                axes[row, col_group*2].set_title(f'{calendar.month_name[month_num]}\nERA5', fontsize=10); axes[row, col_group*2].grid(alpha=0.3)
                axes[row, col_group*2].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                axes[row, col_group*2].set_ylabel(f'{unit}')
                
                conus_times = pd.to_datetime(conus_month[time_dim].values)
                axes[row, col_group*2+1].plot(conus_times, conus_month.mean(dim=[d for d in conus_month.dims if d != time_dim], skipna=True), 'r-')
                axes[row, col_group*2+1].set_title(f'{calendar.month_name[month_num]}\nCONUS', fontsize=10); axes[row, col_group*2+1].grid(alpha=0.3)
                axes[row, col_group*2+1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            except KeyError: continue
        plt.suptitle(f'{era_var} vs {conus_var} Monthly Time Series ({era_year})', fontsize=20, fontweight='bold', y=0.98)
        plt.subplots_adjust(wspace=0.3, hspace=0.6, left=0.05, right=0.95, top=0.92, bottom=0.05)
        plt.savefig(os.path.join(dirs['base'], f'timeseries_{era_var}_all_months.png'), dpi=300, bbox_inches='tight')
        plt.close()

# --- SEASONAL GENERATION FUNCTIONS ---

def generate_seasonal_statistics(era_ds, conus_ds, era_var, conus_var, dirs):
    import matplotlib.patches as mpatches
    print(f"Processing seasonal stats: {era_var} vs {conus_var}...")
    
    try:
        era_seasonal_data, conus_seasonal_data = load_all_seasonal_data(era_ds, conus_ds, era_var, conus_var)
    except KeyError as e:
        print(f"  Skipping {era_var}/{conus_var}: {e}")
        return

    global_min, global_max = compute_global_limits(era_seasonal_data, conus_seasonal_data)
    time_dim = get_time_dimension(conus_ds)

    # Define season order
    seasons = ["winter", "spring", "summer", "autumn"]

    # Collect cleaned seasonal values
    era_season_vals = []
    conus_season_vals = []
    season_labels = []

    for season in seasons:
        era_season = era_seasonal_data[season]
        conus_season = conus_seasonal_data[season]

        # Collapse time/spatial dims
        era_dims = [d for d in era_season.dims if d in ['valid_time', 'time']]
        conus_dims = [d for d in conus_season.dims if d == time_dim]

        era_vals = get_clean_values(era_season.mean(dim=era_dims, skipna=True))
        conus_vals = get_clean_values(conus_season.mean(dim=conus_dims, skipna=True))

        era_season_vals.append(era_vals)
        conus_season_vals.append(conus_vals)
        season_labels.append(season.capitalize())

        # --------------------------
        # Plot all seasons in one boxplot
        # --------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35  # width of each box
        positions_era = [i - width/2 for i in range(1, 5)]
        positions_conus = [i + width/2 for i in range(1, 5)]

        # ERA boxes
        b_era = ax.boxplot(era_season_vals, positions=positions_era, widths=width, patch_artist=True,
                        showfliers=True, medianprops=dict(color='black'))
        for patch in b_era['boxes']:
            patch.set_facecolor('skyblue')

        # CONUS boxes
        b_conus = ax.boxplot(conus_season_vals, positions=positions_conus, widths=width, patch_artist=True,
                            showfliers=True, medianprops=dict(color='black'))
        for patch in b_conus['boxes']:
            patch.set_facecolor('salmon')

        # X-axis labels at the center of each season pair
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([s.capitalize() for s in seasons])

        ax.set_ylabel(era_var)
        ax.set_title(f'Seasonal Box Plot — {era_var}', fontsize=16, fontweight='bold')
        ax.grid(alpha=0.3)

        # Legend
        era_patch = mpatches.Patch(facecolor='skyblue', label='ERA')
        conus_patch = mpatches.Patch(facecolor='salmon', label='CONUS')
        ax.legend(handles=[era_patch, conus_patch])

        plt.tight_layout()
        output_file = os.path.join(dirs['stats'], f'stats_{era_var}_seasonal_box.png')
        plt.savefig(output_file, dpi=300)
        plt.close()

        # --------------------------
        # ECDF Figure
        # --------------------------
        fig_ecdf, axes_ecdf = plt.subplots(2, 2, figsize=(12, 10))
        axes_ecdf = axes_ecdf.flatten()

        for i, season in enumerate(seasons):
            plot_ecdf(axes_ecdf[i], era_season_vals[i], conus_season_vals[i],
                    global_min, global_max, title=f'{season.capitalize()} ECDF')
            if i == 0:  # add legend only once
                axes_ecdf[i].legend(['ERA', 'CONUS'])

        fig_ecdf.suptitle(f'Seasonal ECDF — {era_var}', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0,0,1,0.95])
        output_file_ecdf = os.path.join(dirs['stats'], f'seasonal_ecdf_{era_var}.png')
        plt.savefig(output_file_ecdf, dpi=300)
        plt.close()

        # --------------------------
        # Q-Q Figure (2x2)
        # --------------------------
        fig_qq, axes_qq = plt.subplots(2, 2, figsize=(12, 10))
        axes_qq = axes_qq.flatten()

        for i, season in enumerate(seasons):
            plot_qq(axes_qq[i], era_season_vals[i], conus_season_vals[i],
                    'ERA', 'CONUS', global_min, global_max, title=f'{season.capitalize()} Q-Q')
            if i == 0:  # add legend only once
                axes_qq[i].legend(['ERA', 'CONUS'])

        fig_qq.suptitle(f'Seasonal Q-Q — {era_var}', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0,0,1,0.95])
        output_file_qq = os.path.join(dirs['stats'], f'seasonal_qq_{era_var}.png')
        plt.savefig(output_file_qq, dpi=300)
        plt.close()


def generate_seasonal_maps(era_ds, conus_ds, era_var, conus_var, dirs):
    print(f"Processing seasonal maps: {era_var} vs {conus_var}...")

    time_dim, (lat_name, lon_name) = get_time_dimension(conus_ds), get_coordinate_names(conus_ds)
    
    # Calculate GLOBAL limits first
    all_temps = []
    for month in range(1, 13):
        era_dims = [d for d in era_ds['t2m'].dims if d in ['valid_time', 'time']]
        era_month = trim_to_us(era_ds['t2m'].sel(valid_time=era_ds.valid_time.dt.month == month).mean(dim=era_dims), LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        conus_month = trim_to_us(conus_ds['T2'].sel({time_dim: conus_ds[time_dim].dt.month == month}).mean(dim=time_dim), LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
        all_temps.extend([float(era_month.min()), float(era_month.max()), float(conus_month.min()), float(conus_month.max())])
    vmin, vmax = min(all_temps), max(all_temps)

    seasons = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11]
    }

    # Plot seasonal maps
    for season_name, months in seasons.items():
        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)

        # ERA seasonal mean
        era_dims = [d for d in era_ds[era_var].dims if d in ['valid_time', 'time']]
        era_season = trim_to_us(
            era_ds[era_var].sel(valid_time=era_ds.valid_time.dt.month.isin(months)).mean(dim=era_dims),
            LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
        )

        ax1 = fig.add_subplot(gs[0], projection=create_map_axis())
        ax1.pcolormesh(
            era_season['longitude'], era_season['latitude'], era_season,
            transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto'
        )
        add_map_features(ax1, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        ax1.set_title(f'ERA5 {era_var}', fontsize=12)

        # CONUS seasonal mean
        conus_season = conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month.isin(months)}).mean(dim=time_dim)
        if lat_name in conus_ds and lon_name in conus_ds:
            conus_season = conus_season.assign_coords({lat_name: conus_ds[lat_name], lon_name: conus_ds[lon_name]})
        conus_season = trim_to_us(
            conus_season, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
            lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name]
        )

        ax2 = fig.add_subplot(gs[1], projection=create_map_axis())
        p2 = ax2.pcolormesh(
            conus_season[lon_name], conus_season[lat_name], conus_season,
            transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto'
        )
        add_map_features(ax2, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        ax2.set_title(f'CONUS404 {conus_var}', fontsize=12)

        # Colorbar
        cbar = fig.colorbar(p2, cax=fig.add_subplot(gs[2]), extend='both')
        cbar.set_label(f'{era_var} / {conus_var} units', fontsize=10)

        plt.suptitle(f'{season_name} {era_var} vs {conus_var} Comparison', fontsize=16, fontweight='bold', y=0.95)
        output_file = os.path.join(dirs['maps'], f'map_{era_var}_{season_name.lower()}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
def generate_seasonal_timeseries(era_ds, conus_ds, era_var, conus_var, dirs):
    print(f"Processing seasonal timeseries: {era_var} vs {conus_var}...")

    time_dim = get_time_dimension(conus_ds)
    try: era_year = pd.to_datetime(era_ds.valid_time.values[0]).year
    except: era_year = "Unknown Year"

    seasons = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11]
    }

    for season_name, months in seasons.items():
        try:
            # Select seasonal data
            era_season = era_ds[era_var].sel(valid_time=era_ds.valid_time.dt.month.isin(months))
            conus_season = conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month.isin(months)})

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # ERA5 + CONUS together
            era_times = pd.to_datetime(era_season.valid_time.values)
            reduce_dims = [d for d in era_season.dims if d != 'valid_time']
            axes[0].plot(era_times, era_season.mean(dim=reduce_dims, skipna=True), 'b-', linewidth=1.5, label='ERA5')

            conus_times = pd.to_datetime(conus_season[time_dim].values)
            spatial_dims = [d for d in conus_season.dims if d != time_dim]
            axes[0].plot(conus_times, conus_season.mean(dim=spatial_dims, skipna=True), 'r-', linewidth=1.5, label='CONUS')
            axes[0].set_title(f'{season_name} Time Series', fontsize=12)
            axes[0].grid(alpha=0.3)
            axes[0].legend()

            # Set x-axis ticks at the start of each month
            start_year = era_times[0].year
            months_ticks = pd.date_range(f'{start_year}-01-01', f'{start_year}-12-01', freq='MS')
            axes[0].set_xticks(months_ticks)
            axes[0].set_xticklabels([calendar.month_abbr[m.month] for m in months_ticks])

            # Difference (ERA5 - CONUS)
            # Align times if necessary
            common_times = pd.to_datetime(
                np.intersect1d(era_season.valid_time.values, conus_season[time_dim].values)
            )
            if len(common_times) > 0:
                era_common = era_season.sel(valid_time=common_times)
                conus_common = conus_season.sel({time_dim: common_times})
                axes[1].plot(common_times,
                             era_common.mean(dim=reduce_dims, skipna=True) - conus_common.mean(dim=spatial_dims, skipna=True),
                             'k-', linewidth=1.5)
            axes[1].set_title('Difference (ERA5 - CONUS)', fontsize=12)
            axes[1].grid(alpha=0.3)
            axes[1].set_xticks(months_ticks)
            axes[1].set_xticklabels([calendar.month_abbr[m.month] for m in months_ticks])

            plt.suptitle(f'{era_var} Seasonal Time Series', fontsize=16, fontweight='bold', y=0.98)
            plt.savefig(os.path.join(dirs['timeseries'], f'timeseries_{era_var}_{season_name.lower()}.png'), dpi=300)
            plt.close()
        except KeyError:
            continue

# --- YEARLY GENERATION FUNCTIONS ---

def generate_yearly_single_variable(era_ds, conus_ds, era_var, conus_var, yearly_base_dir):
    """Generates Stats, Maps, and Timeseries for the FULL YEAR for a single variable."""
    print(f"Processing YEARLY data for {era_var}...")
    
    var_dir = os.path.join(yearly_base_dir, era_var)
    Path(var_dir).mkdir(exist_ok=True)

    time_dim = get_time_dimension(conus_ds)
    lat_name, lon_name = get_coordinate_names(conus_ds)
    unit = VARIABLE_UNITS.get(era_var, '')
    
    # Prepare Yearly Aggregates
    era_dims = [d for d in era_ds[era_var].dims if d in ['valid_time', 'time']]
    
    if era_var == 'tp':
        era_yearly_agg = era_ds[era_var].sum(dim=era_dims, skipna=True) * 1000
        conus_yearly_agg = conus_ds[conus_var].sum(dim=time_dim, skipna=True)
    else:
        era_yearly_agg = era_ds[era_var].mean(dim=era_dims, skipna=True)
        conus_yearly_agg = conus_ds[conus_var].mean(dim=time_dim, skipna=True)
    
    if lat_name in conus_ds and lon_name in conus_ds:
        conus_yearly_agg = conus_yearly_agg.assign_coords({lat_name: conus_ds[lat_name], lon_name: conus_ds[lon_name]})
    
    # Trim
    era_yearly_agg = trim_to_us(era_yearly_agg, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    conus_yearly_agg = trim_to_us(conus_yearly_agg, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
    
    # Flatten for Stats
    era_vals = get_clean_values(era_yearly_agg)
    conus_vals = get_clean_values(conus_yearly_agg)
    
    gmin, gmax = min(era_vals.min(), conus_vals.min()), max(era_vals.max(), conus_vals.max())
    
    # Stats Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_box(axes[0], era_vals, conus_vals, ['ERA', 'C404'], gmin, gmax, title='Yearly Box Plot', ylabel=f'{era_var} ({unit})')
    plot_ecdf(axes[1], era_vals, conus_vals, gmin, gmax, title='Yearly ECDF', unit_label=unit)
    plot_qq(axes[2], era_vals, conus_vals, 'ERA', 'C404', gmin, gmax, title='Yearly Q-Q Plot', unit_label=unit)
    plt.suptitle(f'Yearly Statistics - {era_var}', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(var_dir, f'stats_{era_var}_yearly.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Map Plot
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
    
    ax1 = fig.add_subplot(gs[0], projection=create_map_axis())
    ax1.pcolormesh(era_yearly_agg['longitude'], era_yearly_agg['latitude'], era_yearly_agg, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=gmin, vmax=gmax, shading='auto')
    add_map_features(ax1, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    ax1.set_title('ERA5 Yearly Agg', fontsize=12)
    
    ax2 = fig.add_subplot(gs[1], projection=create_map_axis())
    p2 = ax2.pcolormesh(conus_yearly_agg[lon_name], conus_yearly_agg[lat_name], conus_yearly_agg, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=gmin, vmax=gmax, shading='auto')
    add_map_features(ax2, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    ax2.set_title('CONUS404 Yearly Agg', fontsize=12)
    
    cbar = fig.colorbar(p2, cax=fig.add_subplot(gs[2]), extend='both')
    cbar.set_label(f'{era_var} ({unit})', fontsize=10)
    plt.suptitle(f'Yearly Agg Map - {era_var}', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(var_dir, f'map_{era_var}_yearly.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Time Series Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    era_times = pd.to_datetime(era_ds.valid_time.values)
    era_ts = era_ds[era_var].mean(dim=[d for d in era_ds[era_var].dims if d != 'valid_time'], skipna=True)
    if era_var == 'tp': era_ts = era_ts * 1000 # Convert for TS
    
    axes[0].plot(era_times, era_ts, 'b-', linewidth=1)
    axes[0].set_title('ERA5 Yearly TS'); axes[0].grid(alpha=0.3)
    axes[0].set_ylabel(f'{unit}')
    
    conus_times = pd.to_datetime(conus_ds[get_time_dimension(conus_ds)].values)
    conus_ts = conus_ds[conus_var].mean(dim=[d for d in conus_ds[conus_var].dims if d != get_time_dimension(conus_ds)], skipna=True)
    axes[1].plot(conus_times, conus_ts, 'r-', linewidth=1)
    axes[1].set_title('CONUS404 Yearly TS'); axes[1].grid(alpha=0.3)
    axes[1].set_ylabel(f'{unit}')
    
    plt.suptitle(f'Yearly Time Series - {era_var}', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(var_dir, f'timeseries_{era_var}_yearly.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_yearly_combined_summary(era_ds, conus_ds, variable_pairs, yearly_base_dir):
    """Creates summary plots combining ALL variables onto single figures."""
    print("Generating Yearly Combined Summary for all variables...")
    
    time_dim = get_time_dimension(conus_ds)
    lat_name, lon_name = get_coordinate_names(conus_ds)
    
    num_vars = len(variable_pairs)
    fig_stats, axes_stats = plt.subplots(num_vars, 3, figsize=(16, 5 * num_vars))
    if num_vars == 1: axes_stats = np.array([axes_stats])
    
    fig_maps = plt.figure(figsize=(15, 6 * num_vars))
    gs_maps = gridspec.GridSpec(num_vars, 3, width_ratios=[1, 1, 0.05], wspace=0.1, hspace=0.3)
    
    fig_ts, axes_ts = plt.subplots(num_vars, 2, figsize=(12, 5 * num_vars))
    if num_vars == 1: axes_ts = np.array([axes_ts])

    for i, (era_var, conus_var) in enumerate(variable_pairs.items()):
        era_dims = [d for d in era_ds[era_var].dims if d in ['valid_time', 'time']]
        unit = VARIABLE_UNITS.get(era_var, '')
        
        # CHECK: Sum & Convert for TP
        if era_var == 'tp':
            era_mean = trim_to_us(era_ds[era_var].sum(dim=era_dims, skipna=True) * 1000, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            conus_mean = conus_ds[conus_var].sum(dim=time_dim, skipna=True)
        else:
            era_mean = trim_to_us(era_ds[era_var].mean(dim=era_dims, skipna=True), LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            conus_mean = conus_ds[conus_var].mean(dim=time_dim, skipna=True)

        if lat_name in conus_ds: conus_mean = conus_mean.assign_coords({lat_name: conus_ds[lat_name], lon_name: conus_ds[lon_name]})
        conus_mean = trim_to_us(conus_mean, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
        
        e_vals, c_vals = get_clean_values(era_mean), get_clean_values(conus_mean)
        gmin, gmax = min(e_vals.min(), c_vals.min()), max(e_vals.max(), c_vals.max())
        
        plot_box(axes_stats[i, 0], e_vals, c_vals, ['ERA', 'C404'], gmin, gmax, title=f'{era_var} Box', ylabel=f'{era_var} ({unit})')
        plot_ecdf(axes_stats[i, 1], e_vals, c_vals, gmin, gmax, title=f'{era_var} ECDF', unit_label=unit)
        plot_qq(axes_stats[i, 2], e_vals, c_vals, 'ERA', 'C404', gmin, gmax, title=f'{era_var} QQ', unit_label=unit)
        
        ax_m1 = fig_maps.add_subplot(gs_maps[i, 0], projection=create_map_axis())
        ax_m1.pcolormesh(era_mean['longitude'], era_mean['latitude'], era_mean, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=gmin, vmax=gmax, shading='auto')
        add_map_features(ax_m1, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX); ax_m1.set_title(f'ERA5 {era_var}')
        
        ax_m2 = fig_maps.add_subplot(gs_maps[i, 1], projection=create_map_axis())
        p2 = ax_m2.pcolormesh(conus_mean[lon_name], conus_mean[lat_name], conus_mean, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=gmin, vmax=gmax, shading='auto')
        add_map_features(ax_m2, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX); ax_m2.set_title(f'CONUS {conus_var}')
        
        fig_maps.colorbar(p2, cax=fig_maps.add_subplot(gs_maps[i, 2]), extend='both').set_label(f'{era_var} ({unit})')
        
        era_ts = era_ds[era_var].mean(dim=[d for d in era_ds[era_var].dims if d != 'valid_time'], skipna=True)
        conus_ts = conus_ds[conus_var].mean(dim=[d for d in conus_ds[conus_var].dims if d != time_dim], skipna=True)
        if era_var == 'tp': era_ts = era_ts * 1000

        axes_ts[i, 0].plot(pd.to_datetime(era_ds.valid_time.values), era_ts, 'b-'); axes_ts[i, 0].set_title(f'ERA5 {era_var} TS')
        axes_ts[i, 0].set_ylabel(f'{unit}')
        axes_ts[i, 1].plot(pd.to_datetime(conus_ds[time_dim].values), conus_ts, 'r-'); axes_ts[i, 1].set_title(f'CONUS {conus_var} TS')
        axes_ts[i, 1].set_ylabel(f'{unit}')

    fig_stats.suptitle('Yearly Stats Summary (All Vars)', fontweight='bold'); fig_stats.savefig(os.path.join(yearly_base_dir, 'summary_stats_yearly.png'), dpi=300, bbox_inches='tight'); plt.close(fig_stats)
    fig_maps.suptitle('Yearly Maps Summary (All Vars)', fontweight='bold'); fig_maps.savefig(os.path.join(yearly_base_dir, 'summary_maps_yearly.png'), dpi=300, bbox_inches='tight'); plt.close(fig_maps)
    fig_ts.suptitle('Yearly Time Series Summary (All Vars)', fontweight='bold'); fig_ts.tight_layout(); fig_ts.savefig(os.path.join(yearly_base_dir, 'summary_timeseries_yearly.png'), dpi=300, bbox_inches='tight'); plt.close(fig_ts)

def main():
    print("="*40); print("ERA5 vs CONUS404 Comparison"); print("="*40)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    yearly_dir = setup_yearly_directories(OUTPUT_DIR)
    
    era_ds, conus_ds = load_datasets(ERA5_FILE, CONUS_FILE)
    
    # 1. Monthly Maps FIRST (especially for t2m)
    if 't2m' in VARIABLE_PAIRS:
        dirs_t2m = setup_directories(OUTPUT_DIR, 't2m', SEPARATE_IMAGES)
        generate_monthly_temperature_maps(era_ds, conus_ds, dirs_t2m, SEPARATE_IMAGES)
    
    # 2. Then Monthly Stats and Timeseries
    for era_var, conus_var in VARIABLE_PAIRS.items():
        dirs = setup_directories(OUTPUT_DIR, era_var, SEPARATE_IMAGES)
        generate_monthly_statistics_plots(era_ds, conus_ds, era_var, conus_var, dirs, SEPARATE_IMAGES)
        generate_monthly_timeseries(era_ds, conus_ds, era_var, conus_var, dirs, SEPARATE_IMAGES)
        generate_yearly_single_variable(era_ds, conus_ds, era_var, conus_var, yearly_dir)

    # 4 Seasonal
    for era_var, conus_var in VARIABLE_PAIRS.items():
        dirs = setup_directories(OUTPUT_DIR, era_var, SEPARATE_IMAGES)
        generate_seasonal_statistics(era_ds, conus_ds, era_var, conus_var, dirs)
        generate_seasonal_timeseries(era_ds, conus_ds, era_var, conus_var, dirs)
    
    # 3. Combined Yearly Summary (All Variables)
    generate_yearly_combined_summary(era_ds, conus_ds, VARIABLE_PAIRS, yearly_dir)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
