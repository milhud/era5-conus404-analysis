#!/usr/local/other/GEOSpyD/24.3.0-0/2024-08-29/envs/py3.12/bin/python3

from setup import load_datasets, setup_directories
from monthly import generate_monthly_statistics_plots,generate_monthly_maps,generate_monthly_timeseries
from seasonal import generate_seasonal_maps,generate_seasonal_statistics,generate_seasonal_timeseries
from yearly import generate_yearly_single_variable, setup_yearly_directories

from pathlib import Path
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
ERA5_FILE = 'era5_2010.nc'
CONUS_FILE = 'conus404_yearly_2010.nc'
OUTPUT_DIR = 'comparison_plots'

# ERA5 (key) : CONUS404 (value)
VARIABLE_PAIRS = {
   # 't2m': 'T2',
   # 'd2m': 'TD2',
   # 'sp': 'PSFC',
   # 'u10': 'U10',
   # 'v10': 'V10',
    'lai': 'LAI',
   # 'tp': 'ACRAINLSM',
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

SEPARATE_IMAGES = True 



# --- MAIN EXECUTION ---

def main():
    print("Starting Comparison...")
    era_ds, conus_ds = load_datasets(ERA5_FILE, CONUS_FILE)
    
    # 1. Process Monthly/Seasonal Data
    for era_var, conus_var in VARIABLE_PAIRS.items():
        print(f"\nProcessing {era_var} vs {conus_var}...")
        dirs = setup_directories(OUTPUT_DIR, era_var, SEPARATE_IMAGES)
        
        # Monthly Stats & Maps (As originally requested)
        generate_monthly_statistics_plots(era_ds, conus_ds, era_var, conus_var, dirs)
        generate_monthly_maps(era_ds, conus_ds, era_var, conus_var, dirs)
        
        # Monthly Timeseries - NEW OVERLAY VERSION
        generate_monthly_timeseries(era_ds, conus_ds, era_var, conus_var, dirs)
        
        # Seasonal Analysis
        generate_seasonal_statistics(era_ds, conus_ds, era_var, conus_var, dirs)
        generate_seasonal_maps(era_ds, conus_ds, era_var, conus_var, dirs)
        generate_seasonal_timeseries(era_ds, conus_ds, era_var, conus_var, dirs)

    # 2. Process Yearly Aggregate Data
    yearly_base = os.path.join(OUTPUT_DIR, 'yearly')
    Path(yearly_base).mkdir(parents=True, exist_ok=True)
    for era_var, conus_var in VARIABLE_PAIRS.items():
        generate_yearly_single_variable(era_ds, conus_ds, era_var, conus_var, yearly_base)

    print("\nComparison Complete.")

if __name__ == "__main__":
    main()

# def main():
#     print("="*40); print("ERA5 vs CONUS404 Comparison"); print("="*40)
#     Path(OUTPUT_DIR).mkdir(exist_ok=True)
#     yearly_dir = setup_yearly_directories(OUTPUT_DIR)
    
#     era_ds, conus_ds = load_datasets(ERA5_FILE, CONUS_FILE)
    
#     # 1. Monthly Maps FIRST (especially for t2m)
#     if 't2m' in VARIABLE_PAIRS:
#         dirs_t2m = setup_directories(OUTPUT_DIR, 't2m', SEPARATE_IMAGES)
#         generate_monthly_temperature_maps(era_ds, conus_ds, dirs_t2m, SEPARATE_IMAGES)
    
#     # 2. Then Monthly Stats and Timeseries
#     for era_var, conus_var in VARIABLE_PAIRS.items():
#         dirs = setup_directories(OUTPUT_DIR, era_var, SEPARATE_IMAGES)
#         generate_monthly_statistics_plots(era_ds, conus_ds, era_var, conus_var, dirs, SEPARATE_IMAGES)
#         generate_monthly_timeseries(era_ds, conus_ds, era_var, conus_var, dirs, SEPARATE_IMAGES)
#         generate_yearly_single_variable(era_ds, conus_ds, era_var, conus_var, yearly_dir)

#     # 4 Seasonal
#     for era_var, conus_var in VARIABLE_PAIRS.items():
#         dirs = setup_directories(OUTPUT_DIR, era_var, SEPARATE_IMAGES)
#         generate_seasonal_statistics(era_ds, conus_ds, era_var, conus_var, dirs)
#         generate_seasonal_timeseries(era_ds, conus_ds, era_var, conus_var, dirs)
    
#     # 3. Combined Yearly Summary (All Variables)
#     generate_yearly_combined_summary(era_ds, conus_ds, VARIABLE_PAIRS, yearly_dir)
    
#     print("\nDone.")

# if __name__ == "__main__":
#     main()
