# era5-conus404-analysis

This script compares ERA5 reanalysis data against CONUS404 model output. It processes NetCDF files to generate statistical visualizations, spatial maps, and time series analysis for specified variables (e.g., Temperature).

### Workflow

The script begins by loading the ERA5 and CONUS404 NetCDF datasets defined in the configuration. It automatically trims both datasets to a specified geographical bounding box covering the United States to ensure consistent spatial comparisons.

Once the data is preprocessed, the script performs two main analytical routines:

Monthly Analysis: It iterates through each month of the year to calculate statistics (ECDF, Box plots, Q-Q plots), generate spatial temperature maps, and plot time series data.

Yearly Aggregation: It calculates yearly mean fields to generate aggregated summary plots for all variables side-by-side.

All resulting visualizations are automatically organized and saved into the output directory.

### Directory Structure

The script automatically creates the following folder structure:

comparison_plots/
├── [variable_name]/       (e.g., t2m)
│   ├── stats/             (Monthly Box, ECDF, Q-Q plots)
│   ├── maps/              (Monthly Spatial Maps)
│   └── timeseries/        (Monthly Time Series plots)
└── yearly/
├── [variable_name]/   (Yearly aggregate plots for specific variable)
└── [summary_plots]    (Combined plots containing all variables)

### Plot Descriptions

Box Plot:
Displays the distribution of data. The box represents the interquartile range (25th to 75th percentile), and the line inside is the median.

ECDF (Empirical Cumulative Distribution Function):
Shows the proportion of data points less than or equal to a specific value. Used to compare the cumulative probabilities of the two datasets.

Q-Q Plot (Quantile-Quantile):
Plots the quantiles of ERA5 against CONUS404. Points following the straight line indicate that the datasets share a similar distribution.

Spatial Maps:
Side-by-side geographic heatmaps showing the average value of the variable over the US region for a given time period.

Time Series:
Line graphs tracking the spatially averaged value of the variable over time.

### Usage

Ensure the NetCDF files are in the same directory or update the file paths in the Configuration section of the script.

Install the required dependencies:

pip install -r requirements.txt


### Run the script:

python3 script.py

Check the 'comparison_plots' directory for output.
