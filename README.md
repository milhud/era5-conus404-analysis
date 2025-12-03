# era5-conus404-analysis
ERA5 vs CONUS404 Comparison Script

This script compares ERA5 reanalysis data against CONUS404 model output. It processes NetCDF files to generate statistical visualizations, spatial maps, and time series analysis for specified variables (e.g., Temperature).

Workflow

[ era5_2015.nc ]       [ conus404_yearly_2010.nc ]
|                          |
v                          v
[ Load Datasets ] -> [ Trim to US Region ]
|
+---------------+---------------+
|                               |
[ Monthly Loop ]               [ Yearly Aggregation ]
|                               |
1. Calculate Stats              1. Calculate Mean Fields
2. Generate Plots               2. Generate Summary Plots
|                               |
v                               v
[ Output: comparison_plots/ ]   [ Output: comparison_plots/yearly/ ]

Directory Structure

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

Run the script using Python 3.

Check the 'comparison_plots' directory for output.
