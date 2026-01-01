from setup import get_time_dimension,get_coordinate_names,trim_to_us,get_clean_values,compute_global_limits
from plotting import create_map_axis,add_map_features,plot_box,plot_ecdf,plot_qq

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import os
import pandas as pd
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

def load_all_seasonal_data(era_ds, conus_ds, era_var, conus_var):
    time_dim = get_time_dimension(conus_ds)
    lat_name, lon_name = get_coordinate_names(conus_ds)
    
    seasons = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11]
    }
    
    era_seasonal_data = {}
    conus_seasonal_data = {}
    
    for season_name, months in seasons.items():
        # ERA5 seasonal data
        era_season = era_ds[era_var].sel(valid_time=era_ds.valid_time.dt.month.isin(months))
        era_seasonal_data[season_name] = era_season
        
        # CONUS seasonal data
        conus_season = conus_ds[conus_var].sel({time_dim: conus_ds[time_dim].dt.month.isin(months)})
        conus_seasonal_data[season_name] = conus_season
    
    return era_seasonal_data, conus_seasonal_data

def generate_seasonal_timeseries(era_ds, conus_ds, era_var, conus_var, dirs):
    """Generate seasonal timeseries with robust time dimension handling."""
    print(f"Processing seasonal timeseries: {era_var} vs {conus_var}...")

    # Explicit Time Dim Detection
    era_time_dim = 'valid_time' if 'valid_time' in era_ds else 'time'
    conus_time_dim = 'Time' if 'Time' in conus_ds else 'time'
    
    lat_name, lon_name = get_coordinate_names(conus_ds)
    unit = VARIABLE_UNITS.get(era_var, '')
    
    seasons = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11]
    }

    for season_name, months in seasons.items():
        # --- ERA5 ---
        era_seasonal = era_ds[era_var].sel({era_time_dim: era_ds[era_time_dim].dt.month.isin(months)})
        era_trimmed = trim_to_us(era_seasonal, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        
        era_reduce_dims = [d for d in era_trimmed.dims if d != era_time_dim]
        era_ts = era_trimmed.mean(dim=era_reduce_dims)
        
        era_times = pd.to_datetime(era_ts[era_time_dim].values)
        era_values = era_ts.values
        if era_values.ndim > 1: era_values = era_values.squeeze()
        
        # --- CONUS ---
        conus_seasonal = conus_ds[conus_var].sel({conus_time_dim: conus_ds[conus_time_dim].dt.month.isin(months)})
        conus_trimmed = trim_to_us(
            conus_seasonal, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
            lat_grid=conus_ds[lat_name], lon_grid=conus_ds[lon_name]
        )
        
        conus_reduce_dims = [d for d in conus_trimmed.dims if d != conus_time_dim]
        conus_ts = conus_trimmed.mean(dim=conus_reduce_dims)

        conus_times = pd.to_datetime(conus_ts[conus_time_dim].values)
        conus_values = conus_ts.values
        if conus_values.ndim > 1: conus_values = conus_values.squeeze()
        
        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(era_times, era_values, 'o-', linewidth=2, markersize=2, 
                label='ERA5', color='#2E86AB', alpha=0.8)
        ax.plot(conus_times, conus_values, 's-', linewidth=2, markersize=2, 
                label='CONUS404', color='#A23B72', alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{era_var} ({unit})', fontsize=12, fontweight='bold')
        ax.set_title(f'{season_name} Timeseries: {era_var} vs {conus_var}', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        output_file = os.path.join(dirs['timeseries'], f'timeseries_{era_var}_{season_name.lower()}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
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

    seasons = ["winter", "spring", "summer", "autumn"]
    era_season_vals = []
    conus_season_vals = []

    for season in seasons:
        era_season = era_seasonal_data[season]
        conus_season = conus_seasonal_data[season]

        era_dims = [d for d in era_season.dims if d in ['valid_time', 'time']]
        conus_dims = [d for d in conus_season.dims if d == time_dim]

        era_vals = get_clean_values(era_season.mean(dim=era_dims, skipna=True))
        conus_vals = get_clean_values(conus_season.mean(dim=conus_dims, skipna=True))

        era_season_vals.append(era_vals)
        conus_season_vals.append(conus_vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    positions_era = [i - width/2 for i in range(1, 5)]
    positions_conus = [i + width/2 for i in range(1, 5)]

    b_era = ax.boxplot(era_season_vals, positions=positions_era, widths=width, patch_artist=True,
                        showfliers=True, medianprops=dict(color='black'))
    for patch in b_era['boxes']: patch.set_facecolor('skyblue')

    b_conus = ax.boxplot(conus_season_vals, positions=positions_conus, widths=width, patch_artist=True,
                            showfliers=True, medianprops=dict(color='black'))
    for patch in b_conus['boxes']: patch.set_facecolor('salmon')

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([s.capitalize() for s in seasons])
    ax.set_ylabel(era_var)
    ax.set_title(f'Seasonal Box Plot — {era_var}', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3)

    era_patch = mpatches.Patch(facecolor='skyblue', label='ERA')
    conus_patch = mpatches.Patch(facecolor='salmon', label='CONUS')
    ax.legend(handles=[era_patch, conus_patch])

    plt.tight_layout()
    output_file = os.path.join(dirs['stats'], f'stats_{era_var}_seasonal_box.png')
    plt.savefig(output_file, dpi=300)
    plt.close()