#!/usr/local/other/GEOSpyD/24.3.0-0/2024-08-29/envs/py3.12/bin/python3

import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ks_2samp
import os
from pathlib import Path
import warnings
import logging
import sys

warnings.filterwarnings("ignore")

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("metrics_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_OUTPUT_DIR = "comparison_metrics"

CONUS_BASE = "../../final_data/conus404_yearly_{year}.nc"
ERA5_BASE = "../../../sduan/pipeline/data/processed/era5_{year}.nc"

YEARS = range(1980, 2021)

VARIABLE_PAIRS = {
    "t2m": "T2",
    "d2m": "TD2",
    "sp": "PSFC",
    "u10": "U10",
    "v10": "V10",
    "lai": "LAI",
    "tp": "PREC_ACC_NC",
}

VARIABLE_UNITS = {
    "t2m": "K",
    "d2m": "K",
    "sp": "hPa",
    "u10": "m/s",
    "v10": "m/s",
    "lai": "Index",
    "tp": "mm",
}

VARIABLE_NAMES = {
    "t2m": "2m Temperature",
    "d2m": "2m Dewpoint",
    "sp": "Surface Pressure",
    "u10": "10m U-Wind",
    "v10": "10m V-Wind",
    "lai": "Leaf Area Index",
    "tp": "Precipitation",
    "wind_speed": "10m Wind Speed",
}

LAT_MIN, LAT_MAX = 24, 50
LON_MIN, LON_MAX = -125, -66

SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
}

PERCENTILES = [50, 75, 90, 95, 99]

PRECIP_WET_THRESHOLD = 1.0       # mm/day
PRECIP_HEAVY_THRESHOLD = 10.0    # mm/day
PRECIP_VERY_HEAVY_THRESHOLD = 20.0  # mm/day

PRECIP_EXTREME_PERCENTILES = [95, 99]

WIND_THRESHOLDS = [5.0, 10.0, 15.0, 20.0]

# Maximum number of spatial samples used for the KS test.
# This prevents the enormous CONUS404 grid from completely
# dominating the statistical test.
MAX_KS_SAMPLES = 100000

RANDOM_SEED = 42


# ============================================================
# DATASET HELPERS
# ============================================================

def get_time_dimension(ds):

    if "Time" in ds.dims:
        return "Time"

    if "time" in ds.dims:
        return "time"

    if "valid_time" in ds.dims:
        return "valid_time"

    raise ValueError(
        f"Could not identify time dimension. "
        f"Dimensions are: {list(ds.dims)}"
    )


def get_time_coordinate(ds):

    if "valid_time" in ds.coords:
        return "valid_time"

    if "time" in ds.coords:
        return "time"

    if "Time" in ds.coords:
        return "Time"

    time_dim = get_time_dimension(ds)

    return time_dim


def get_coordinate_names(ds):

    if "XLAT" in ds:
        lat_name = "XLAT"
    elif "latitude" in ds:
        lat_name = "latitude"
    elif "lat" in ds:
        lat_name = "lat"
    else:
        raise ValueError(
            "Could not identify latitude coordinate."
        )

    if "XLONG" in ds:
        lon_name = "XLONG"
    elif "longitude" in ds:
        lon_name = "longitude"
    elif "lon" in ds:
        lon_name = "lon"
    else:
        raise ValueError(
            "Could not identify longitude coordinate."
        )

    return lat_name, lon_name


def load_datasets(era_file, conus_file):

    try:

        era_ds = xr.open_dataset(era_file)
        conus_ds = xr.open_dataset(conus_file)

        logger.info(
            f"Loaded ERA5: {era_file}"
        )

        logger.info(
            f"Loaded CONUS404: {conus_file}"
        )

        return era_ds, conus_ds

    except Exception as e:

        logger.error(
            f"Could not load datasets: {e}"
        )

        return None, None


# ============================================================
# UNIT CONVERSION
# ============================================================

def convert_era_units(data, var_name):

    if var_name == "tp":

        # ERA5 total precipitation is normally meters.
        # Convert to mm.
        return data * 1000.0

    if var_name == "sp":

        # Pa -> hPa
        return data / 100.0

    return data


def convert_conus_units(data, era_var):

    if era_var == "sp":

        # Pa -> hPa
        return data / 100.0

    return data


# ============================================================
# CONUS MASKING
# ============================================================

def get_us_mask(
    lat,
    lon,
    lat_min=LAT_MIN,
    lat_max=LAT_MAX,
    lon_min=LON_MIN,
    lon_max=LON_MAX
):

    return (
        (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
    )


def trim_era5_to_us(data):

    if (
        "latitude" in data.dims
        and "longitude" in data.dims
    ):

        return data.sel(
            latitude=slice(
                LAT_MAX,
                LAT_MIN
            ),
            longitude=slice(
                LON_MIN,
                LON_MAX
            )
        )

    return data


def trim_conus_to_us(
    data,
    conus_ds
):

    lat_name, lon_name = get_coordinate_names(
        conus_ds
    )

    lat = conus_ds[lat_name]
    lon = conus_ds[lon_name]

    mask = get_us_mask(
        lat,
        lon
    )

    return data.where(
        mask,
        drop=True
    )


# ============================================================
# CLEAN ARRAY
# ============================================================

def clean_pair(reference, model):

    reference = np.asarray(
        reference
    ).flatten()

    model = np.asarray(
        model
    ).flatten()

    valid = (
        np.isfinite(reference)
        & np.isfinite(model)
    )

    return (
        reference[valid],
        model[valid]
    )


def clean_array(values):

    values = np.asarray(
        values
    ).flatten()

    return values[
        np.isfinite(values)
    ]


# ============================================================
# KS SAMPLING
# ============================================================

def sample_for_ks(
    reference,
    model,
    max_samples=MAX_KS_SAMPLES,
    seed=RANDOM_SEED
):

    reference = clean_array(reference)
    model = clean_array(model)

    rng = np.random.default_rng(seed)

    if len(reference) > max_samples:

        reference = rng.choice(
            reference,
            size=max_samples,
            replace=False
        )

    if len(model) > max_samples:

        model = rng.choice(
            model,
            size=max_samples,
            replace=False
        )

    return reference, model


# ============================================================
# GENERAL DISTRIBUTIONAL METRICS
# ============================================================

def calculate_distribution_metrics(
    reference,
    model
):

    reference = clean_array(reference)
    model = clean_array(model)

    if len(reference) == 0 or len(model) == 0:

        return {
            "n_era5": 0,
            "n_conus404": 0,
            "era5_mean": np.nan,
            "conus404_mean": np.nan,
            "mean_difference": np.nan,
            "era5_std": np.nan,
            "conus404_std": np.nan,
            "std_ratio": np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
        }

    # --------------------------------------------------------
    # KS test
    # --------------------------------------------------------

    era_ks, conus_ks = sample_for_ks(
        reference,
        model
    )

    ks_result = ks_2samp(
        era_ks,
        conus_ks,
        alternative="two-sided",
        method="auto"
    )

    # --------------------------------------------------------
    # Basic distribution statistics
    # --------------------------------------------------------

    era_mean = np.mean(reference)
    conus_mean = np.mean(model)

    era_std = np.std(reference)
    conus_std = np.std(model)

    if era_std > 0:

        std_ratio = (
            conus_std / era_std
        )

    else:

        std_ratio = np.nan

    metrics = {

        "n_era5": len(reference),

        "n_conus404": len(model),

        "era5_mean": era_mean,

        "conus404_mean": conus_mean,

        "mean_difference":
            conus_mean - era_mean,

        "era5_std": era_std,

        "conus404_std": conus_std,

        "std_ratio": std_ratio,

        "ks_statistic":
            ks_result.statistic,

        "ks_pvalue":
            ks_result.pvalue,
    }

    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    for percentile in PERCENTILES:

        era_q = np.percentile(
            reference,
            percentile
        )

        conus_q = np.percentile(
            model,
            percentile
        )

        difference = (
            conus_q - era_q
        )

        if era_q != 0:

            percent_bias = (
                100.0
                * difference
                / abs(era_q)
            )

        else:

            percent_bias = np.nan

        prefix = f"p{percentile}"

        metrics[
            f"{prefix}_era5"
        ] = era_q

        metrics[
            f"{prefix}_conus404"
        ] = conus_q

        metrics[
            f"{prefix}_difference"
        ] = difference

        metrics[
            f"{prefix}_percent_bias"
        ] = percent_bias

    return metrics


# ============================================================
# PAIRED METRICS
# ============================================================

def calculate_paired_metrics(
    reference,
    model
):

    reference, model = clean_pair(
        reference,
        model
    )

    if len(reference) == 0:

        return {}

    error = (
        model - reference
    )

    bias = np.mean(error)

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    reference_std = np.std(
        reference
    )

    # nRMSE normalized by reference standard deviation
    if reference_std > 0:

        nrmse = (
            rmse / reference_std
        )

    else:

        nrmse = np.nan

    # Pearson correlation
    if (
        len(reference) > 1
        and np.std(reference) > 0
        and np.std(model) > 0
    ):

        r, pvalue = pearsonr(
            reference,
            model
        )

        r_squared = r ** 2

    else:

        r = np.nan
        pvalue = np.nan
        r_squared = np.nan

    return {

        "n": len(reference),

        "era5_mean":
            np.mean(reference),

        "conus404_mean":
            np.mean(model),

        "bias":
            bias,

        "mae":
            mae,

        "rmse":
            rmse,

        "pearson_r":
            r,

        "pearson_pvalue":
            pvalue,

        "r_squared":
            r_squared,

        "nrmse":
            nrmse,
    }


# ============================================================
# TEMPORAL SPATIAL MEANS
# ============================================================

def get_spatial_mean_timeseries(
    ds,
    var_name,
    is_era5,
    conus_ds=None
):

    data = ds[var_name]

    time_dim = get_time_dimension(
        ds
    )

    # --------------------------------------------------------
    # Spatial dimensions
    # --------------------------------------------------------

    spatial_dims = [
        d for d in data.dims
        if d != time_dim
    ]

    data = data.mean(
        dim=spatial_dims,
        skipna=True
    )

    # --------------------------------------------------------
    # Units
    # --------------------------------------------------------

    if is_era5:

        data = convert_era_units(
            data,
            var_name
        )

    else:

        data = convert_conus_units(
            data,
            var_name
        )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    time_coord = get_time_coordinate(
        ds
    )

    dates = pd.to_datetime(
        data[time_coord].values
    )

    values = np.asarray(
        data.values
    ).squeeze()

    return pd.DataFrame({
        "date": dates,
        "value": values
    })


# ============================================================
# DAILY PRECIPITATION
# ============================================================

def get_daily_precipitation(
    ds,
    var_name,
    is_era5,
    conus_ds=None
):

    data = ds[var_name]

    time_dim = get_time_dimension(
        ds
    )

    # --------------------------------------------------------
    # Convert precipitation units
    # --------------------------------------------------------

    if is_era5:

        data = data * 1000.0

    # CONUS units are assumed to already be mm.
    # If your CONUS variable is in another unit,
    # change this here.

    # --------------------------------------------------------
    # Spatial mean first
    # --------------------------------------------------------

    spatial_dims = [
        d for d in data.dims
        if d != time_dim
    ]

    data = data.mean(
        dim=spatial_dims,
        skipna=True
    )

    # --------------------------------------------------------
    # Convert to daily totals if sub-daily
    # --------------------------------------------------------

    time_coord = get_time_coordinate(
        ds
    )

    dates = pd.to_datetime(
        data[time_coord].values
    )

    if len(dates) > 1:

        median_delta = np.median(
            np.diff(dates)
            / np.timedelta64(1, "h")
        )

    else:

        median_delta = 24.0

    # If data are hourly/sub-daily,
    # sum to daily precipitation.
    if median_delta < 20:

        data = data.resample(
            {time_dim: "1D"}
        ).sum(
            skipna=True
        )

    dates = pd.to_datetime(
        data[time_coord].values
    )

    values = np.asarray(
        data.values
    ).squeeze()

    return pd.DataFrame({
        "date": dates,
        "precip": values
    })


# ============================================================
# PRECIPITATION METRICS
# ============================================================

def calculate_precipitation_metrics(
    era_precip,
    conus_precip
):

    era = np.asarray(
        era_precip
    ).flatten()

    conus = np.asarray(
        conus_precip
    ).flatten()

    valid = (
        np.isfinite(era)
        & np.isfinite(conus)
    )

    era = era[valid]
    conus = conus[valid]

    if len(era) == 0:

        return {}

    metrics = {}

    # --------------------------------------------------------
    # Wet days
    # --------------------------------------------------------

    era_wet = (
        era >= PRECIP_WET_THRESHOLD
    )

    conus_wet = (
        conus >= PRECIP_WET_THRESHOLD
    )

    era_wet_frequency = (
        100.0
        * np.mean(era_wet)
    )

    conus_wet_frequency = (
        100.0
        * np.mean(conus_wet)
    )

    metrics[
        "era5_wet_day_frequency"
    ] = era_wet_frequency

    metrics[
        "conus404_wet_day_frequency"
    ] = conus_wet_frequency

    metrics[
        "wet_day_frequency_difference"
    ] = (
        conus_wet_frequency
        - era_wet_frequency
    )

    # --------------------------------------------------------
    # Wet-day intensity
    # --------------------------------------------------------

    era_wet_values = era[
        era_wet
    ]

    conus_wet_values = conus[
        conus_wet
    ]

    if len(era_wet_values) > 0:

        era_intensity = np.mean(
            era_wet_values
        )

    else:

        era_intensity = np.nan

    if len(conus_wet_values) > 0:

        conus_intensity = np.mean(
            conus_wet_values
        )

    else:

        conus_intensity = np.nan

    metrics[
        "era5_mean_wet_day_intensity"
    ] = era_intensity

    metrics[
        "conus404_mean_wet_day_intensity"
    ] = conus_intensity

    metrics[
        "wet_day_intensity_difference"
    ] = (
        conus_intensity
        - era_intensity
    )

    if (
        np.isfinite(era_intensity)
        and era_intensity != 0
    ):

        metrics[
            "wet_day_intensity_percent_bias"
        ] = (
            100.0
            * (
                conus_intensity
                - era_intensity
            )
            / abs(era_intensity)
        )

    else:

        metrics[
            "wet_day_intensity_percent_bias"
        ] = np.nan

    # --------------------------------------------------------
    # Total precipitation
    # --------------------------------------------------------

    era_total = np.sum(
        era
    )

    conus_total = np.sum(
        conus
    )

    metrics[
        "era5_total_precip"
    ] = era_total

    metrics[
        "conus404_total_precip"
    ] = conus_total

    metrics[
        "total_precip_difference"
    ] = (
        conus_total
        - era_total
    )

    if era_total != 0:

        metrics[
            "total_precip_percent_bias"
        ] = (
            100.0
            * (
                conus_total
                - era_total
            )
            / abs(era_total)
        )

    else:

        metrics[
            "total_precip_percent_bias"
        ] = np.nan

    # --------------------------------------------------------
    # SDII
    # --------------------------------------------------------

    metrics[
        "era5_sdii"
    ] = era_intensity

    metrics[
        "conus404_sdii"
    ] = conus_intensity

    metrics[
        "sdii_difference"
    ] = (
        conus_intensity
        - era_intensity
    )

    # --------------------------------------------------------
    # R10mm
    # --------------------------------------------------------

    era_r10 = np.sum(
        era >= PRECIP_HEAVY_THRESHOLD
    )

    conus_r10 = np.sum(
        conus >= PRECIP_HEAVY_THRESHOLD
    )

    metrics[
        "era5_r10mm"
    ] = era_r10

    metrics[
        "conus404_r10mm"
    ] = conus_r10

    metrics[
        "r10mm_difference"
    ] = (
        conus_r10
        - era_r10
    )

    # --------------------------------------------------------
    # R20mm
    # --------------------------------------------------------

    era_r20 = np.sum(
        era >= PRECIP_VERY_HEAVY_THRESHOLD
    )

    conus_r20 = np.sum(
        conus >= PRECIP_VERY_HEAVY_THRESHOLD
    )

    metrics[
        "era5_r20mm"
    ] = era_r20

    metrics[
        "conus404_r20mm"
    ] = conus_r20

    metrics[
        "r20mm_difference"
    ] = (
        conus_r20
        - era_r20
    )

    # --------------------------------------------------------
    # RX1day
    # --------------------------------------------------------

    metrics[
        "era5_rx1day"
    ] = np.max(era)

    metrics[
        "conus404_rx1day"
    ] = np.max(conus)

    metrics[
        "rx1day_difference"
    ] = (
        metrics["conus404_rx1day"]
        - metrics["era5_rx1day"]
    )

    # --------------------------------------------------------
    # RX5day
    # --------------------------------------------------------

    if len(era) >= 5:

        era_rx5 = (
            pd.Series(era)
            .rolling(5)
            .sum()
            .dropna()
            .max()
        )

        conus_rx5 = (
            pd.Series(conus)
            .rolling(5)
            .sum()
            .dropna()
            .max()
        )

    else:

        era_rx5 = np.nan
        conus_rx5 = np.nan

    metrics[
        "era5_rx5day"
    ] = era_rx5

    metrics[
        "conus404_rx5day"
    ] = conus_rx5

    metrics[
        "rx5day_difference"
    ] = (
        conus_rx5
        - era_rx5
    )

    # --------------------------------------------------------
    # R95p / R99p
    # --------------------------------------------------------

    for percentile in PRECIP_EXTREME_PERCENTILES:

        if len(era_wet_values) > 0:

            threshold = np.percentile(
                era_wet_values,
                percentile
            )

            era_extreme_total = np.sum(
                era[
                    era >= threshold
                ]
            )

            conus_extreme_total = np.sum(
                conus[
                    conus >= threshold
                ]
            )

        else:

            threshold = np.nan
            era_extreme_total = np.nan
            conus_extreme_total = np.nan

        prefix = f"r{percentile}p"

        metrics[
            f"era5_{prefix}_threshold"
        ] = threshold

        metrics[
            f"era5_{prefix}"
        ] = era_extreme_total

        metrics[
            f"conus404_{prefix}"
        ] = conus_extreme_total

        metrics[
            f"{prefix}_difference"
        ] = (
            conus_extreme_total
            - era_extreme_total
        )

        if (
            np.isfinite(era_extreme_total)
            and era_extreme_total != 0
        ):

            metrics[
                f"{prefix}_percent_bias"
            ] = (
                100.0
                * (
                    conus_extreme_total
                    - era_extreme_total
                )
                / abs(era_extreme_total)
            )

        else:

            metrics[
                f"{prefix}_percent_bias"
            ] = np.nan

    return metrics


# ============================================================
# WIND SPEED
# ============================================================

def get_wind_timeseries(
    ds,
    u_name,
    v_name,
    is_era5
):

    time_dim = get_time_dimension(
        ds
    )

    u = ds[u_name]
    v = ds[v_name]

    spatial_dims = [
        d for d in u.dims
        if d != time_dim
    ]

    u = u.mean(
        dim=spatial_dims,
        skipna=True
    )

    spatial_dims_v = [
        d for d in v.dims
        if d != time_dim
    ]

    v = v.mean(
        dim=spatial_dims_v,
        skipna=True
    )

    wind_speed = np.sqrt(
        u ** 2 + v ** 2
    )

    time_coord = get_time_coordinate(
        ds
    )

    dates = pd.to_datetime(
        wind_speed[time_coord].values
    )

    values = np.asarray(
        wind_speed.values
    ).squeeze()

    return pd.DataFrame({
        "date": dates,
        "wind_speed": values
    })


def calculate_wind_metrics(
    era_wind,
    conus_wind
):

    era, conus = clean_pair(
        era_wind,
        conus_wind
    )

    if len(era) == 0:
        return {}

    metrics = calculate_paired_metrics(
        era,
        conus
    )

    distribution_metrics = (
        calculate_distribution_metrics(
            era,
            conus
        )
    )

    metrics.update(
        distribution_metrics
    )

    # --------------------------------------------------------
    # Wind-speed percentiles
    # --------------------------------------------------------

    for percentile in PERCENTILES:

        era_q = np.percentile(
            era,
            percentile
        )

        conus_q = np.percentile(
            conus,
            percentile
        )

        difference = (
            conus_q - era_q
        )

        if era_q != 0:

            percent_bias = (
                100.0
                * difference
                / abs(era_q)
            )

        else:

            percent_bias = np.nan

        prefix = f"wind_p{percentile}"

        metrics[
            f"{prefix}_era5"
        ] = era_q

        metrics[
            f"{prefix}_conus404"
        ] = conus_q

        metrics[
            f"{prefix}_difference"
        ] = difference

        metrics[
            f"{prefix}_percent_bias"
        ] = percent_bias

    # --------------------------------------------------------
    # High-wind frequency
    # --------------------------------------------------------

    for threshold in WIND_THRESHOLDS:

        era_frequency = (
            100.0
            * np.mean(
                era >= threshold
            )
        )

        conus_frequency = (
            100.0
            * np.mean(
                conus >= threshold
            )
        )

        name = (
            f"{threshold:g}ms"
        )

        metrics[
            f"era5_wind_ge_{name}_frequency"
        ] = era_frequency

        metrics[
            f"conus404_wind_ge_{name}_frequency"
        ] = conus_frequency

        metrics[
            f"wind_ge_{name}_frequency_difference"
        ] = (
            conus_frequency
            - era_frequency
        )

    return metrics


# ============================================================
# PROCESS ONE VARIABLE / ONE YEAR
# ============================================================

def process_variable(
    era_ds,
    conus_ds,
    era_var,
    conus_var,
    year
):

    results = []

    if era_var not in era_ds:

        logger.warning(
            f"{era_var} missing from ERA5"
        )

        return results

    if conus_var not in conus_ds:

        logger.warning(
            f"{conus_var} missing from CONUS404"
        )

        return results

    # ========================================================
    # SPATIAL DISTRIBUTION
    # ========================================================

    try:

        era_data = (
            era_ds[era_var]
            .mean(
                dim=[
                    d for d in era_ds[era_var].dims
                    if d in ["time", "valid_time"]
                ],
                skipna=True
            )
        )

        era_data = trim_era5_to_us(
            era_data
        )

        era_data = convert_era_units(
            era_data,
            era_var
        )

        conus_data = (
            conus_ds[conus_var]
            .mean(
                dim=[
                    d for d in conus_ds[conus_var].dims
                    if d == get_time_dimension(conus_ds)
                ],
                skipna=True
            )
        )

        conus_data = trim_conus_to_us(
            conus_data,
            conus_ds
        )

        conus_data = convert_conus_units(
            conus_data,
            era_var
        )

        distribution = (
            calculate_distribution_metrics(
                era_data.values,
                conus_data.values
            )
        )

        distribution.update({

            "year": year,

            "variable": era_var,

            "variable_name":
                VARIABLE_NAMES.get(
                    era_var,
                    era_var
                ),

            "units":
                VARIABLE_UNITS.get(
                    era_var,
                    ""
                ),

            "metric_type":
                "native_spatial_distribution",

            "resolution_note":
                "Native grids; not paired pixelwise"
        })

        results.append(
            distribution
        )

    except Exception as e:

        logger.error(
            f"Spatial distribution failed "
            f"{era_var} {year}: {e}"
        )

    # ========================================================
    # TEMPORAL METRICS
    # ========================================================

    try:

        era_ts = get_spatial_mean_timeseries(
            era_ds,
            era_var,
            is_era5=True
        )

        conus_ts = get_spatial_mean_timeseries(
            conus_ds,
            conus_var,
            is_era5=False
        )

        merged = pd.merge(
            era_ts,
            conus_ts,
            on="date",
            how="inner",
            suffixes=(
                "_era5",
                "_conus404"
            )
        )

        if len(merged) > 1:

            paired = calculate_paired_metrics(
                merged["value_era5"].values,
                merged["value_conus404"].values
            )

            paired.update(
                calculate_distribution_metrics(
                    merged["value_era5"].values,
                    merged["value_conus404"].values
                )
            )

            paired.update({

                "year": year,

                "variable": era_var,

                "variable_name":
                    VARIABLE_NAMES.get(
                        era_var,
                        era_var
                    ),

                "units":
                    VARIABLE_UNITS.get(
                        era_var,
                        ""
                    ),

                "metric_type":
                    "temporal_spatial_mean"
            })

            results.append(
                paired
            )

    except Exception as e:

        logger.error(
            f"Temporal metrics failed "
            f"{era_var} {year}: {e}"
        )

    return results


# ============================================================
# SEASONAL TEMPORAL METRICS
# ============================================================

def calculate_seasonal_metrics(
    era_ds,
    conus_ds,
    era_var,
    conus_var,
    year
):

    results = []

    try:

        era_ts = get_spatial_mean_timeseries(
            era_ds,
            era_var,
            is_era5=True
        )

        conus_ts = get_spatial_mean_timeseries(
            conus_ds,
            conus_var,
            is_era5=False
        )

        merged = pd.merge(
            era_ts,
            conus_ts,
            on="date",
            how="inner",
            suffixes=(
                "_era5",
                "_conus404"
            )
        )

        merged["month"] = (
            merged["date"].dt.month
        )

        for season, months in SEASONS.items():

            season_df = merged[
                merged["month"].isin(months)
            ]

            if len(season_df) < 2:
                continue

            metrics = calculate_paired_metrics(
                season_df[
                    "value_era5"
                ].values,

                season_df[
                    "value_conus404"
                ].values
            )

            distribution = (
                calculate_distribution_metrics(
                    season_df[
                        "value_era5"
                    ].values,

                    season_df[
                        "value_conus404"
                    ].values
                )
            )

            metrics.update(
                distribution
            )

            metrics.update({

                "year": year,

                "season": season,

                "variable": era_var,

                "variable_name":
                    VARIABLE_NAMES.get(
                        era_var,
                        era_var
                    ),

                "units":
                    VARIABLE_UNITS.get(
                        era_var,
                        ""
                    )
            })

            results.append(
                metrics
            )

    except Exception as e:

        logger.error(
            f"Seasonal metrics failed "
            f"{era_var} {year}: {e}"
        )

    return results


# ============================================================
# PRECIPITATION PROCESSING
# ============================================================

def process_precipitation(
    era_ds,
    conus_ds,
    year
):

    try:

        era = get_daily_precipitation(
            era_ds,
            "tp",
            is_era5=True
        )

        conus = get_daily_precipitation(
            conus_ds,
            "PREC_ACC_NC",
            is_era5=False
        )

        merged = pd.merge(
            era,
            conus,
            on="date",
            how="inner",
            suffixes=(
                "_era5",
                "_conus404"
            )
        )

        if len(merged) == 0:

            logger.warning(
                f"No matched precipitation dates "
                f"for {year}"
            )

            return []

        metrics = calculate_paired_metrics(
            merged[
                "precip_era5"
            ].values,

            merged[
                "precip_conus404"
            ].values
        )

        distribution = (
            calculate_distribution_metrics(
                merged[
                    "precip_era5"
                ].values,

                merged[
                    "precip_conus404"
                ].values
            )
        )

        precip_specific = (
            calculate_precipitation_metrics(
                merged[
                    "precip_era5"
                ].values,

                merged[
                    "precip_conus404"
                ].values
            )
        )

        metrics.update(
            distribution
        )

        metrics.update(
            precip_specific
        )

        metrics.update({

            "year": year,

            "variable": "tp",

            "variable_name":
                "Precipitation",

            "units":
                "mm/day",

            "metric_type":
                "precipitation",

            "n_days":
                len(merged)
        })

        return [metrics]

    except Exception as e:

        logger.error(
            f"Precipitation failed "
            f"for {year}: {e}"
        )

        return []


# ============================================================
# WIND PROCESSING
# ============================================================

def process_wind(
    era_ds,
    conus_ds,
    year
):

    try:

        era = get_wind_timeseries(
            era_ds,
            "u10",
            "v10",
            is_era5=True
        )

        conus = get_wind_timeseries(
            conus_ds,
            "U10",
            "V10",
            is_era5=False
        )

        merged = pd.merge(
            era,
            conus,
            on="date",
            how="inner",
            suffixes=(
                "_era5",
                "_conus404"
            )
        )

        if len(merged) == 0:
            return []

        metrics = calculate_wind_metrics(
            merged[
                "wind_speed_era5"
            ].values,

            merged[
                "wind_speed_conus404"
            ].values
        )

        metrics.update({

            "year": year,

            "variable":
                "wind_speed",

            "variable_name":
                "10m Wind Speed",

            "units":
                "m/s",

            "metric_type":
                "derived_wind_speed"
        })

        return [metrics]

    except Exception as e:

        logger.error(
            f"Wind metrics failed "
            f"for {year}: {e}"
        )

        return []


# ============================================================
# PROCESS ONE YEAR
# ============================================================

def process_year(year):

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        f"PROCESSING {year}"
    )
    logger.info("=" * 70)

    era_file = ERA5_BASE.format(
        year=year
    )

    conus_file = CONUS_BASE.format(
        year=year
    )

    if not os.path.exists(era_file):

        logger.warning(
            f"ERA5 file missing: {era_file}"
        )

        return [], [], [], []

    if not os.path.exists(conus_file):

        logger.warning(
            f"CONUS404 file missing: {conus_file}"
        )

        return [], [], [], []

    era_ds, conus_ds = load_datasets(
        era_file,
        conus_file
    )

    if era_ds is None or conus_ds is None:

        return [], [], [], []

    spatial_results = []
    seasonal_results = []
    precip_results = []
    wind_results = []

    # ========================================================
    # GENERAL VARIABLES
    # ========================================================

    for era_var, conus_var in VARIABLE_PAIRS.items():

        if era_var == "tp":
            continue

        if era_var in ["u10", "v10"]:
            # They are handled individually plus derived wind.
            pass

        results = process_variable(
            era_ds,
            conus_ds,
            era_var,
            conus_var,
            year
        )

        for result in results:

            if result.get(
                "metric_type"
            ) == "native_spatial_distribution":

                spatial_results.append(
                    result
                )

            elif result.get(
                "metric_type"
            ) == "temporal_spatial_mean":

                spatial_results.append(
                    result
                )

        # Seasonal
        seasonal = calculate_seasonal_metrics(
            era_ds,
            conus_ds,
            era_var,
            conus_var,
            year
        )

        seasonal_results.extend(
            seasonal
        )

    # ========================================================
    # PRECIPITATION
    # ========================================================

    precip_results.extend(
        process_precipitation(
            era_ds,
            conus_ds,
            year
        )
    )

    # ========================================================
    # DERIVED WIND SPEED
    # ========================================================

    wind_results.extend(
        process_wind(
            era_ds,
            conus_ds,
            year
        )
    )

    era_ds.close()
    conus_ds.close()

    return (
        spatial_results,
        seasonal_results,
        precip_results,
        wind_results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "ERA5 vs CONUS404 QUANTITATIVE VALIDATION"
    )
    logger.info("=" * 70)

    Path(
        BASE_OUTPUT_DIR
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    all_spatial = []
    all_seasonal = []
    all_precip = []
    all_wind = []

    successful_years = []
    failed_years = []

    # ========================================================
    # PROCESS YEARS
    # ========================================================

    for year in YEARS:

        try:

            (
                spatial,
                seasonal,
                precip,
                wind
            ) = process_year(year)

            if (
                spatial
                or seasonal
                or precip
                or wind
            ):

                successful_years.append(
                    year
                )

            else:

                failed_years.append(
                    year
                )

            all_spatial.extend(
                spatial
            )

            all_seasonal.extend(
                seasonal
            )

            all_precip.extend(
                precip
            )

            all_wind.extend(
                wind
            )

        except Exception as e:

            logger.error(
                f"Year {year} failed: {e}"
            )

            failed_years.append(
                year
            )

    # ========================================================
    # SAVE SPATIAL/TEMPORAL RESULTS
    # ========================================================

    if all_spatial:

        df = pd.DataFrame(
            all_spatial
        )

        file = os.path.join(
            BASE_OUTPUT_DIR,
            "annual_spatial_metrics.csv"
        )

        df.to_csv(
            file,
            index=False,
            float_format="%.6f"
        )

        logger.info(
            f"Saved: {file}"
        )

    # ========================================================
    # SAVE SEASONAL RESULTS
    # ========================================================

    if all_seasonal:

        df = pd.DataFrame(
            all_seasonal
        )

        file = os.path.join(
            BASE_OUTPUT_DIR,
            "seasonal_temporal_metrics.csv"
        )

        df.to_csv(
            file,
            index=False,
            float_format="%.6f"
        )

        logger.info(
            f"Saved: {file}"
        )

    # ========================================================
    # SAVE PRECIPITATION
    # ========================================================

    if all_precip:

        df = pd.DataFrame(
            all_precip
        )

        file = os.path.join(
            BASE_OUTPUT_DIR,
            "precipitation_metrics.csv"
        )

        df.to_csv(
            file,
            index=False,
            float_format="%.6f"
        )

        logger.info(
            f"Saved: {file}"
        )

    # ========================================================
    # SAVE WIND
    # ========================================================

    if all_wind:

        df = pd.DataFrame(
            all_wind
        )

        file = os.path.join(
            BASE_OUTPUT_DIR,
            "wind_speed_metrics.csv"
        )

        df.to_csv(
            file,
            index=False,
            float_format="%.6f"
        )

        logger.info(
            f"Saved: {file}"
        )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    summary_tables = []

    if all_spatial:

        spatial_df = pd.DataFrame(
            all_spatial
        )

        # Average metrics over years.
        numeric_columns = [
            "bias",
            "mae",
            "rmse",
            "pearson_r",
            "r_squared",
            "nrmse",
            "ks_statistic",
            "ks_pvalue",
            "mean_difference",
            "std_ratio",
            "p50_difference",
            "p75_difference",
            "p90_difference",
            "p95_difference",
            "p99_difference",
            "p50_percent_bias",
            "p75_percent_bias",
            "p90_percent_bias",
            "p95_percent_bias",
            "p99_percent_bias",
        ]

        available = [
            c for c in numeric_columns
            if c in spatial_df.columns
        ]

        if available:

            summary = (
                spatial_df
                .groupby(
                    [
                        "variable",
                        "variable_name",
                        "metric_type"
                    ]
                )[available]
                .mean()
                .reset_index()
            )

            summary_tables.append(
                summary
            )

    if all_precip:

        precip_df = pd.DataFrame(
            all_precip
        )

        numeric_columns = [
            "bias",
            "mae",
            "rmse",
            "pearson_r",
            "r_squared",
            "nrmse",
            "ks_statistic",
            "ks_pvalue",
            "wet_day_frequency_difference",
            "wet_day_intensity_difference",
            "wet_day_intensity_percent_bias",
            "total_precip_percent_bias",
            "r10mm_difference",
            "r20mm_difference",
            "rx1day_difference",
            "rx5day_difference",
            "r95p_difference",
            "r95p_percent_bias",
            "r99p_difference",
            "r99p_percent_bias",
        ]

        available = [
            c for c in numeric_columns
            if c in precip_df.columns
        ]

        if available:

            summary = (
                precip_df[available]
                .mean()
                .to_frame()
                .T
            )

            summary[
                "variable"
            ] = "tp"

            summary[
                "variable_name"
            ] = "Precipitation"

            summary[
                "metric_type"
            ] = "precipitation"

            summary_tables.append(
                summary
            )

    if all_wind:

        wind_df = pd.DataFrame(
            all_wind
        )

        numeric_columns = [
            "bias",
            "mae",
            "rmse",
            "pearson_r",
            "r_squared",
            "nrmse",
            "ks_statistic",
            "ks_pvalue",
            "wind_p95_difference",
            "wind_p99_difference",
            "wind_p95_percent_bias",
            "wind_p99_percent_bias",
        ]

        available = [
            c for c in numeric_columns
            if c in wind_df.columns
        ]

        if available:

            summary = (
                wind_df[available]
                .mean()
                .to_frame()
                .T
            )

            summary[
                "variable"
            ] = "wind_speed"

            summary[
                "variable_name"
            ] = "10m Wind Speed"

            summary[
                "metric_type"
            ] = "derived_wind_speed"

            summary_tables.append(
                summary
            )

    if summary_tables:

        summary_df = pd.concat(
            summary_tables,
            ignore_index=True,
            sort=False
        )

        summary_file = os.path.join(
            BASE_OUTPUT_DIR,
            "summary_metrics.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False,
            float_format="%.6f"
        )

        logger.info(
            f"Saved: {summary_file}"
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)

    logger.info(
        f"Successful years: "
        f"{len(successful_years)}"
    )

    logger.info(
        f"Failed/skipped years: "
        f"{len(failed_years)}"
    )

    if failed_years:

        logger.info(
            f"Failed years: {failed_years}"
        )

    logger.info(
        f"Output directory: "
        f"{BASE_OUTPUT_DIR}"
    )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()