from setup import get_time_dimension,get_coordinate_names,trim_to_us,get_clean_values,compute_global_limits
from plotting import create_map_axis,add_map_features,plot_box,plot_ecdf,plot_qq

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd
import cartopy.crs as ccrs
import os
from pathlib import Path
import calendar

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


def generate_monthly_temperature_maps(era_ds, conus_ds, era_var, conus_var, dirs):
    print(f"Processing monthly maps: {era_var} vs {conus_var}...")
    if era_var not in era_ds or conus_var not in conus_ds: return
    time_dim, (lat_name, lon_name) = get_time_dimension(conus_ds), get_coordinate_names(conus_ds)
    unit = VARIABLE_UNITS.get(era_var)
    
    all_temps = []
    for month in range(1, 13):
        era_dims = [d for d in era_ds[era_var].dims if d in ['valid_time', 'time']]
        era_month = trim_to_us(era_ds[era_var].sel(valid_time=era_ds.valid_time.dt.month == month).mean(dim=era_dims), LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        conus_month = trim_to_us(conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month == month}).mean(dim=time_dim), LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
        all_temps.extend([float(era_month.min()), float(era_month.max()), float(conus_month.min()), float(conus_month.max())])
    vmin, vmax = min(all_temps), max(all_temps)

    for month_idx in range(12):
        month_num = month_idx + 1
        month_name = calendar.month_name[month_num]
        
        era_dims = [d for d in era_ds[era_var].dims if d in ['valid_time', 'time']]
        era_m = trim_to_us(era_ds[era_var].sel(valid_time=era_ds.valid_time.dt.month == month_num).mean(dim=era_dims), LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        conus_m = conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month == month_num}).mean(dim=time_dim)
        if lat_name in conus_ds and lon_name in conus_ds: conus_m = conus_m.assign_coords({lat_name: conus_ds[lat_name], lon_name: conus_ds[lon_name]})
        conus_m = trim_to_us(conus_m, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
        
        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
        
        ax1 = fig.add_subplot(gs[0], projection=create_map_axis())
        ax1.pcolormesh(era_m['longitude'], era_m['latitude'], era_m, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
        add_map_features(ax1, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        ax1.set_title('ERA5', fontsize=12)
        
        ax2 = fig.add_subplot(gs[1], projection=create_map_axis())
        p2 = ax2.pcolormesh(conus_m[lon_name], conus_m[lat_name], conus_m, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
        add_map_features(ax2, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        ax2.set_title('CONUS404', fontsize=12)
        
        cbar = fig.colorbar(p2, cax=fig.add_subplot(gs[2]), extend='both')
        cbar.set_label(f'{era_var} ({unit})', fontsize=10)
        plt.suptitle(f'{month_name} {era_var} Comparison', fontsize=16, fontweight='bold', y=0.95)
        plt.savefig(os.path.join(dirs['maps'], f'map_{era_var}_month{month_num:02d}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_monthly_statistics_plots(era_ds, conus_ds, era_var, conus_var, dirs):
    print(f"Processing monthly stats: {era_var} vs {conus_var}...")
    try:
        era_monthly_data, conus_monthly_data = load_all_monthly_data(era_ds, conus_ds, era_var, conus_var)
    except KeyError as e:
        print(f"  Skipping {era_var}/{conus_var}: {e}")
        return

    global_min, global_max = compute_global_limits(era_monthly_data, conus_monthly_data, era_var)
    time_dim = get_time_dimension(conus_ds)
    unit = VARIABLE_UNITS.get(era_var, '')
    
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



def generate_monthly_timeseries(era_ds, conus_ds, era_var, conus_var, dirs):
    """Generates monthly timeseries overlays with corrected time dimension handling."""
    print(f"Processing monthly timeseries: {era_var} vs {conus_var}...")
    
    # --- ERA5 Time Detection ---
    # ERA5 usually uses 'valid_time'. We prioritize that.
    if 'valid_time' in era_ds:
        era_time_dim = 'valid_time'
    elif 'time' in era_ds:
        era_time_dim = 'time'
    else:
        era_time_dim = list(era_ds.dims)[0] # Fallback
        
    # --- CONUS Time Detection ---
    # CONUS404 usually uses 'Time' or 'time'
    if 'Time' in conus_ds:
        conus_time_dim = 'Time'
    elif 'time' in conus_ds:
        conus_time_dim = 'time'
    else:
        conus_time_dim = list(conus_ds.dims)[0]

    lat_name, lon_name = get_coordinate_names(conus_ds)
    unit = VARIABLE_UNITS.get(era_var, '')
    
    for month in range(1, 13):
        month_name = calendar.month_name[month]
        
        # --- 1. ERA5 Data ---
        # Robustly select month using the identified time dim
        try:
            era_month = era_ds[era_var].sel({era_time_dim: era_ds[era_time_dim].dt.month == month})
        except AttributeError:
            # Fallback if .dt fails (e.g. not decoded); try standard index selection
            # assuming decoded times, but if not, this block prevents crash
            print(f"Warning: Could not access .dt on {era_time_dim}. Ensure data is decoded.")
            continue

        era_trimmed = trim_to_us(era_month, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        
        # Collapse all non-time dimensions
        era_reduce_dims = [d for d in era_trimmed.dims if d != era_time_dim]
        era_ts = era_trimmed.mean(dim=era_reduce_dims)
        
        era_times = pd.to_datetime(era_ts[era_time_dim].values)
        era_values = era_ts.values
        if era_values.ndim > 1: era_values = era_values.squeeze()

        # --- 2. CONUS Data ---
        conus_month = conus_ds[conus_var].sel({conus_time_dim: conus_ds[conus_time_dim].dt.month == month})
        conus_trimmed = trim_to_us(conus_month, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                                   lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name])
        
        # Collapse all non-time dimensions
        conus_reduce_dims = [d for d in conus_trimmed.dims if d != conus_time_dim]
        conus_ts = conus_trimmed.mean(dim=conus_reduce_dims)
        
        conus_times = pd.to_datetime(conus_ts[conus_time_dim].values)
        conus_values = conus_ts.values
        if conus_values.ndim > 1: conus_values = conus_values.squeeze()

        # --- 3. Plotting ---
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(era_times, era_values, '-', linewidth=2, label='ERA5', color='royalblue', alpha=0.8)
        ax.plot(conus_times, conus_values, '--', linewidth=2, label='CONUS404', color='crimson', alpha=0.8)
        
        ax.set_title(f'{month_name} Timeseries Comparison: {era_var} vs {conus_var}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{era_var} ({unit})', fontsize=12)
        ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.set_xlabel(f'Day of {month_name}')

        plt.tight_layout()
        output_file = os.path.join(dirs['timeseries'], f'timeseries_{era_var}_month{month:02d}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
