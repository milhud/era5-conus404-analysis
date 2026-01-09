#!/usr/local/other/GEOSpyD/24.3.0-0/2024-08-29/envs/py3.12/bin/python3

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress
import os
from pathlib import Path
import warnings
import logging
import sys

warnings.filterwarnings("ignore")

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# configuration
BASE_OUTPUT_DIR = 'comparison_plots'
CONUS_BASE = "../../final_data/conus404_yearly_{year}.nc"
ERA5_BASE = "../../../sduan/pipeline/data/processed/era5_{year}.nc"
YEARS = range(1980, 2021)

# variable pairs: era5 (key) : conus404 (value)
VARIABLE_PAIRS = {
    't2m': 'T2',
    'd2m': 'TD2',
    'sp': 'PSFC',
    'u10': 'U10',
    'v10': 'V10',
    'lai': 'LAI',
    'tp': 'PREC_ACC_NC',
}

# units for labeling
VARIABLE_UNITS = {
    't2m': 'K',
    'd2m': 'K',
    'sp': 'hPa',
    'u10': 'm/s',
    'v10': 'm/s',
    'lai': 'Index',
    'tp': 'mm',
}

# nice names for plotting
VARIABLE_NAMES = {
    't2m': '2m Temperature',
    'd2m': '2m Dewpoint',
    'sp': 'Surface Pressure',
    'u10': '10m U-Wind',
    'v10': '10m V-Wind',
    'lai': 'Leaf Area Index',
    'tp': 'Precipitation',
}

# us bounds
LAT_MIN, LAT_MAX = 24, 50
LON_MIN, LON_MAX = -125, -66

# seasons definition
SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11]
}

# helper functions

def load_datasets(era_file, conus_file):
    try:
        era_ds = xr.open_dataset(era_file)
        conus_ds = xr.open_dataset(conus_file)
        logger.info(f"loaded era5: {era_file}")
        logger.info(f"loaded conus404: {conus_file}")
        return era_ds, conus_ds
    except FileNotFoundError as e:
        logger.error(f"file not found: {e}")
        return None, None

def get_time_dimension(ds):
    if 'Time' in ds.dims:
        return 'Time'
    elif 'time' in ds.dims:
        return 'time'
    elif 'valid_time' in ds.dims:
        return 'valid_time'
    else:
        return list(ds.dims)[0]

def get_coordinate_names(ds):
    # handle wrf coordinates (xlat/xlong) and regular lat/lon
    lat_name = 'XLAT' if 'XLAT' in ds else 'latitude' if 'latitude' in ds else 'lat'
    lon_name = 'XLONG' if 'XLONG' in ds else 'longitude' if 'longitude' in ds else 'lon'
    return lat_name, lon_name

def trim_to_us(data, lat_min, lat_max, lon_min, lon_max, lat_grid=None, lon_grid=None):
    # for wrf/conus404 with 2d coordinates
    if lat_grid is not None and lon_grid is not None:
        mask = (
            (lat_grid >= lat_min) & (lat_grid <= lat_max) &
            (lon_grid >= lon_min) & (lon_grid <= lon_max)
        )
        return data.where(mask, drop=True)
    
    # for era5 with 1d coordinates
    if 'latitude' in data.dims and 'longitude' in data.dims:
        return data.sel(
            latitude=slice(lat_max, lat_min), 
            longitude=slice(lon_min, lon_max)
        )
    return data

def get_clean_values(data):
    vals = data.values.flatten()
    return vals[np.isfinite(vals)]

def convert_units(data, var_name):
    # convert era5 tp from meters to millimeters
    if var_name == 'tp':
        return data * 1000.0
    # convert surface pressure from pascals to hectopascals
    elif var_name == 'sp':
        return data / 100.0
    else:
        return data

def create_map_projection():
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

# data processing functions

def load_seasonal_data(era_ds, conus_ds, era_var, conus_var):
    logger.info(f"  loading seasonal data...")
    
    era_time_dim = 'valid_time' if 'valid_time' in era_ds else 'time'
    conus_time_dim = get_time_dimension(conus_ds)
    lat_name, lon_name = get_coordinate_names(conus_ds)
    
    era_seasonal_data = {}
    conus_seasonal_data = {}
    
    for season_name, months in SEASONS.items():
        # era5 seasonal mean
        era_season = era_ds[era_var].sel(
            {era_time_dim: era_ds[era_time_dim].dt.month.isin(months)}
        )
        era_time_dims = [d for d in era_season.dims if d in ['valid_time', 'time']]
        era_season_mean = era_season.mean(dim=era_time_dims)
        era_season_mean = trim_to_us(era_season_mean, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        era_season_mean = convert_units(era_season_mean, era_var)
        era_seasonal_data[season_name] = era_season_mean
        
        # conus seasonal mean
        conus_season = conus_ds[conus_var].sel(
            {conus_time_dim: conus_ds[conus_time_dim].dt.month.isin(months)}
        )
        conus_season_mean = conus_season.mean(dim=conus_time_dim)
        
        # assign coordinates if needed for wrf data
        if lat_name in conus_ds and lon_name in conus_ds:
            conus_season_mean = conus_season_mean.assign_coords({
                lat_name: conus_ds[lat_name],
                lon_name: conus_ds[lon_name]
            })
        
        conus_season_mean = trim_to_us(
            conus_season_mean, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
            lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name]
        )
        # convert conus pressure from pa to hpa
        if era_var == 'sp':
            conus_season_mean = conus_season_mean / 100.0
        conus_seasonal_data[season_name] = conus_season_mean
    
    return era_seasonal_data, conus_seasonal_data

def compute_yearly_mean(era_ds, conus_ds, era_var, conus_var):
    logger.info(f"  computing yearly means...")
    
    # era5
    era_time_dim = 'valid_time' if 'valid_time' in era_ds else 'time'
    era_time_dims = [d for d in era_ds[era_var].dims if d in ['valid_time', 'time']]
    era_yearly = era_ds[era_var].mean(dim=era_time_dims)
    era_yearly = trim_to_us(era_yearly, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    era_yearly = convert_units(era_yearly, era_var)
    
    # conus
    conus_time_dim = get_time_dimension(conus_ds)
    lat_name, lon_name = get_coordinate_names(conus_ds)
    conus_yearly = conus_ds[conus_var].mean(dim=conus_time_dim)
    
    if lat_name in conus_ds and lon_name in conus_ds:
        conus_yearly = conus_yearly.assign_coords({
            lat_name: conus_ds[lat_name],
            lon_name: conus_ds[lon_name]
        })
    
    conus_yearly = trim_to_us(
        conus_yearly, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name]
    )
    # convert conus pressure from pa to hpa
    if era_var == 'sp':
        conus_yearly = conus_yearly / 100.0
    
    return era_yearly, conus_yearly

def compute_global_limits(era_seasonal_data, conus_seasonal_data):
    all_vals = []
    
    for season in SEASONS.keys():
        era_vals = get_clean_values(era_seasonal_data[season])
        conus_vals = get_clean_values(conus_seasonal_data[season])
        all_vals.extend(era_vals)
        all_vals.extend(conus_vals)
    
    return np.min(all_vals), np.max(all_vals)

# plotting functions

def plot_seasonal_boxplots(era_seasonal_data, conus_seasonal_data, 
                          era_var, output_path):
    logger.info(f"  generating seasonal box plots...")
    
    season_names = list(SEASONS.keys())
    era_season_vals = []
    conus_season_vals = []
    
    for season in season_names:
        era_vals = get_clean_values(era_seasonal_data[season])
        conus_vals = get_clean_values(conus_seasonal_data[season])
        era_season_vals.append(era_vals)
        conus_season_vals.append(conus_vals)
    
    global_min, global_max = compute_global_limits(era_seasonal_data, conus_seasonal_data)
    y_range = global_max - global_min
    y_pad = y_range * 0.1
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    width = 0.35
    positions_era = [i - width/2 for i in range(1, 5)]
    positions_conus = [i + width/2 for i in range(1, 5)]
    
    bp_era = ax.boxplot(
        era_season_vals, 
        positions=positions_era, 
        widths=width,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(facecolor='skyblue', edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )
    
    bp_conus = ax.boxplot(
        conus_season_vals, 
        positions=positions_conus, 
        widths=width,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(facecolor='salmon', edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )
    
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(season_names, fontsize=12)
    ax.set_ylabel(f'{VARIABLE_NAMES.get(era_var, era_var)} ({VARIABLE_UNITS.get(era_var, "")})', 
                  fontsize=13, fontweight='bold')
    ax.set_title(f'Seasonal Comparison: {VARIABLE_NAMES.get(era_var, era_var)}', 
                 fontsize=15, fontweight='bold', pad=15)
    
    ax.set_ylim(global_min - y_pad, global_max + y_pad)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    era_patch = mpatches.Patch(facecolor='skyblue', edgecolor='black', label='ERA5')
    conus_patch = mpatches.Patch(facecolor='salmon', edgecolor='black', label='CONUS404')
    ax.legend(handles=[era_patch, conus_patch], fontsize=11, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"    saved: {output_path}")

def plot_qq_plot(era_seasonal_data, conus_seasonal_data, era_var, output_path):
    logger.info(f"  generating q-q plot...")
    
    era_all = []
    conus_all = []
    
    for season in SEASONS.keys():
        era_all.extend(get_clean_values(era_seasonal_data[season]))
        conus_all.extend(get_clean_values(conus_seasonal_data[season]))
    
    era_all = np.array(era_all)
    conus_all = np.array(conus_all)
    
    n = min(len(era_all), len(conus_all))
    quantiles = np.linspace(0, 1, min(n, 1000))
    era_q = np.quantile(era_all, quantiles)
    conus_q = np.quantile(conus_all, quantiles)
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    ax.scatter(era_q, conus_q, alpha=0.5, s=10, color='darkblue', label='Data')
    
    if len(era_q) > 1 and (era_q.std() > 0 or conus_q.std() > 0):
        slope, intercept, r_value, _, _ = linregress(era_q, conus_q)
        fit_line = slope * era_q + intercept
        ax.plot(era_q, fit_line, 'r-', alpha=0.8, linewidth=2,
                label=f'Fit: R²={r_value**2:.3f}')
    
    lims = [
        min(era_q.min(), conus_q.min()),
        max(era_q.max(), conus_q.max())
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='1:1 Line')
    
    # set axis limits to data range with small padding
    x_range = era_q.max() - era_q.min()
    y_range = conus_q.max() - conus_q.min()
    x_pad = x_range * 0.05
    y_pad = y_range * 0.05
    ax.set_xlim(era_q.min() - x_pad, era_q.max() + x_pad)
    ax.set_ylim(conus_q.min() - y_pad, conus_q.max() + y_pad)
    
    ax.set_xlabel(f'ERA5 Quantiles ({VARIABLE_UNITS.get(era_var, "")})', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'CONUS404 Quantiles ({VARIABLE_UNITS.get(era_var, "")})', 
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Q-Q Plot: {VARIABLE_NAMES.get(era_var, era_var)}', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"    saved: {output_path}")

def plot_yearly_timeseries(era_ds, conus_ds, era_var, conus_var, output_path):
    logger.info(f"  generating yearly timeseries...")
    
    era_time_dim = 'valid_time' if 'valid_time' in era_ds else 'time'
    
    era_data = era_ds[era_var]
    era_spatial_dims = [d for d in era_data.dims if d not in ['valid_time', 'time']]
    era_ts = era_data.mean(dim=era_spatial_dims, skipna=True)
    
    if 'time' in era_ts.dims and era_time_dim == 'valid_time':
        era_ts = era_ts.mean(dim='time')
    
    # apply unit conversion
    era_ts = convert_units(era_ts, era_var)
    
    era_times = pd.to_datetime(era_ts[era_time_dim].values)
    era_values = era_ts.values
    if era_values.ndim > 1:
        era_values = era_values.squeeze()
    
    conus_time_dim = get_time_dimension(conus_ds)
    
    conus_data = conus_ds[conus_var]
    conus_spatial_dims = [d for d in conus_data.dims if d != conus_time_dim]
    conus_ts = conus_data.mean(dim=conus_spatial_dims, skipna=True)
    
    # convert conus pressure from pa to hpa
    if era_var == 'sp':
        conus_ts = conus_ts / 100.0
    
    conus_times = pd.to_datetime(conus_ts[conus_time_dim].values)
    conus_values = conus_ts.values
    if conus_values.ndim > 1:
        conus_values = conus_values.squeeze()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(era_times, era_values, '-', linewidth=2, 
            label='ERA5', color='#2E86AB', alpha=0.8)
    ax.plot(conus_times, conus_values, '--', linewidth=2, 
            label='CONUS404', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{VARIABLE_NAMES.get(era_var, era_var)} ({VARIABLE_UNITS.get(era_var, "")})', 
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Yearly Time Series (Spatial Mean): {VARIABLE_NAMES.get(era_var, era_var)}', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.set_facecolor('#f8f9fa')
    
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"    saved: {output_path}")

def plot_side_by_side_heatmaps(era_yearly, conus_yearly, era_var, conus_var, 
                               conus_ds, output_path):
    logger.info(f"  generating side-by-side heatmaps...")
    
    lat_name, lon_name = get_coordinate_names(conus_ds)
    
    era_vals = get_clean_values(era_yearly)
    conus_vals = get_clean_values(conus_yearly)
    vmin = min(era_vals.min(), conus_vals.min())
    vmax = max(era_vals.max(), conus_vals.max())
    
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], 
                          wspace=0.05, hspace=0)
    
    ax1 = fig.add_subplot(gs[0], projection=create_map_projection())
    im1 = ax1.pcolormesh(
        era_yearly['longitude'], era_yearly['latitude'], era_yearly,
        transform=ccrs.PlateCarree(), cmap='RdYlBu_r', 
        vmin=vmin, vmax=vmax, shading='auto'
    )
    add_map_features(ax1, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    ax1.set_title('ERA5', fontsize=14, fontweight='bold', pad=10)
    
    ax2 = fig.add_subplot(gs[1], projection=create_map_projection())
    im2 = ax2.pcolormesh(
        conus_yearly[lon_name], conus_yearly[lat_name], conus_yearly,
        transform=ccrs.PlateCarree(), cmap='RdYlBu_r', 
        vmin=vmin, vmax=vmax, shading='auto'
    )
    add_map_features(ax2, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    ax2.set_title('CONUS404', fontsize=14, fontweight='bold', pad=10)
    
    cax_right = fig.add_subplot(gs[2])
    
    cbar_right = fig.colorbar(im2, cax=cax_right, extend='both')
    cbar_right.set_label(f'{VARIABLE_UNITS.get(era_var, "")}', 
                         fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Yearly Mean Comparison: {VARIABLE_NAMES.get(era_var, era_var)}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"    saved: {output_path}")

# main processing

def process_variable(era_ds, conus_ds, era_var, conus_var, output_dir):
    logger.info(f"processing: {VARIABLE_NAMES.get(era_var, era_var)} ({era_var} vs {conus_var})")
    
    # check if variables exist in datasets
    if era_var not in era_ds:
        logger.warning(f"variable {era_var} not found in ERA5 dataset, skipping")
        return
    if conus_var not in conus_ds:
        logger.warning(f"variable {conus_var} not found in CONUS404 dataset, skipping")
        return
    
    var_dir = os.path.join(output_dir, era_var)
    Path(var_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        era_seasonal_data, conus_seasonal_data = load_seasonal_data(
            era_ds, conus_ds, era_var, conus_var
        )
        
        era_yearly, conus_yearly = compute_yearly_mean(
            era_ds, conus_ds, era_var, conus_var
        )
        
        plot_seasonal_boxplots(
            era_seasonal_data, conus_seasonal_data, era_var,
            os.path.join(var_dir, f'{era_var}_seasonal_boxplots.png')
        )
        
        plot_qq_plot(
            era_seasonal_data, conus_seasonal_data, era_var,
            os.path.join(var_dir, f'{era_var}_qq_plot.png')
        )
        
        plot_yearly_timeseries(
            era_ds, conus_ds, era_var, conus_var,
            os.path.join(var_dir, f'{era_var}_yearly_timeseries.png')
        )
        
        plot_side_by_side_heatmaps(
            era_yearly, conus_yearly, era_var, conus_var, conus_ds,
            os.path.join(var_dir, f'{era_var}_heatmap_comparison.png')
        )
        
        logger.info(f"completed processing for {era_var}")
        
    except Exception as e:
        logger.error(f"error processing {era_var}: {str(e)}")
        import traceback
        traceback.print_exc()

def process_year(year):
    logger.info(f"\n{'='*70}")
    logger.info(f"processing year {year}")
    logger.info(f"{'='*70}")
    
    era_file = ERA5_BASE.format(year=year)
    conus_file = CONUS_BASE.format(year=year)
    
    if not os.path.exists(era_file):
        logger.warning(f"era5 file not found for {year}: {era_file}")
        return False
    if not os.path.exists(conus_file):
        logger.warning(f"conus404 file not found for {year}: {conus_file}")
        return False
    
    year_output_dir = os.path.join(BASE_OUTPUT_DIR, str(year))
    Path(year_output_dir).mkdir(parents=True, exist_ok=True)
    
    era_ds, conus_ds = load_datasets(era_file, conus_file)
    if era_ds is None or conus_ds is None:
        logger.error(f"failed to load datasets for {year}")
        return False
    
    for era_var, conus_var in VARIABLE_PAIRS.items():
        try:
            process_variable(era_ds, conus_ds, era_var, conus_var, year_output_dir)
        except Exception as e:
            logger.error(f"failed to process {era_var} for year {year}: {str(e)}")
    
    era_ds.close()
    conus_ds.close()
    
    logger.info(f"completed year {year}")
    return True

def main():
    logger.info("\n" + "="*70)
    logger.info("era5 vs conus404 dataset comparison")
    logger.info("="*70 + "\n")
    
    Path(BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    successful_years = []
    failed_years = []
    
    for year in YEARS:
        success = process_year(year)
        if success:
            successful_years.append(year)
        else:
            failed_years.append(year)
    
    logger.info("\n" + "="*70)
    logger.info("processing complete!")
    logger.info(f"output directory: {BASE_OUTPUT_DIR}")
    logger.info(f"successful years ({len(successful_years)}): {successful_years}")
    if failed_years:
        logger.info(f"failed/skipped years ({len(failed_years)}): {failed_years}")
    logger.info("="*70 + "\n")

if __name__ == "__main__":
    main()
