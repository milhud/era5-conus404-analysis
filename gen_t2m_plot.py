#!/usr/local/other/GEOSpyD/24.3.0-0/2024-08-29/envs/py3.12/bin/python3
"""Generate ocean-masked t2m heatmap for a single year."""
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

YEAR = 2015
ERA5_FILE  = f"/gpfsm/dnb33/sduan/pipeline/data/processed/era5_{YEAR}.nc"
CONUS_FILE = f"/gpfsm/dnb33/hpmille1/final_data/conus404_yearly_{YEAR}.nc"
LAT_MIN, LAT_MAX = 24, 50
LON_MIN, LON_MAX = -125, -66

os.makedirs("plots", exist_ok=True)

proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39,
                             standard_parallels=(33, 45))

def _axes_xy(ax, lon, lat):
    """Project (lon, lat) → (x_ax, y_ax) in axes-fraction coordinates."""
    xp, yp = proj.transform_point(lon, lat, ccrs.PlateCarree())
    xd, yd = ax.transData.transform((xp, yp))
    return ax.transAxes.inverted().transform((xd, yd))

def add_map_features(ax, lat_side=None):
    ax.coastlines('50m', color='black', linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=1)
    ax.add_feature(cfeature.LAKES, facecolor='white', zorder=1)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    # gridlines — no auto labels, all placed manually outside the frame
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
    gl.xlocator = matplotlib.ticker.FixedLocator(range(-120, -60, 10))
    gl.ylocator = matplotlib.ticker.FixedLocator(range(25, 55, 5))

    # lon labels: below the axes frame on both panels
    for lon in range(-120, -60, 10):
        try:
            x_ax, _ = _axes_xy(ax, lon, LAT_MIN)
            if -0.05 <= x_ax <= 1.05:
                ax.annotate(f"{abs(lon)}°W",
                            xy=(x_ax, 0), xycoords='axes fraction',
                            xytext=(0, -16), textcoords='offset points',
                            ha='center', va='top', fontsize=7,
                            clip_on=False)
        except Exception:
            pass

    # lat labels: left side for ERA5, right side for CONUS404
    lat_lon_ref = LON_MIN if lat_side == 'left' else LON_MAX
    ha = 'right' if lat_side == 'left' else 'left'
    x_anchor = 0 if lat_side == 'left' else 1
    x_offset = -8 if lat_side == 'left' else 8
    if lat_side in ('left', 'right'):
        for lat in range(25, 55, 5):
            try:
                _, y_ax = _axes_xy(ax, lat_lon_ref, lat)
                if 0.0 <= y_ax <= 1.0:
                    ax.annotate(f"{lat}°N",
                                xy=(x_anchor, y_ax), xycoords='axes fraction',
                                xytext=(x_offset, 0), textcoords='offset points',
                                ha=ha, va='center', fontsize=7,
                                clip_on=False)
            except Exception:
                pass

era   = xr.open_dataset(ERA5_FILE)
conus = xr.open_dataset(CONUS_FILE)

# ── ERA5 t2m yearly mean ─────────────────────────────────────────────────────
era_t2m = era['t2m'].mean(dim=[d for d in era['t2m'].dims
                               if d in ('time', 'valid_time')])
era_t2m = era_t2m.sel(latitude=slice(LAT_MAX, LAT_MIN),
                       longitude=slice(LON_MIN, LON_MAX))

# apply lsm mask (0=ocean → NaN)
lsm = era['lsm'].isel(time=0, valid_time=0).sel(
    latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
era_arr = np.where(lsm.values >= 0.5, era_t2m.values, np.nan)
lat_e = era_t2m['latitude'].values
lon_e = era_t2m['longitude'].values

# ── CONUS404 T2 yearly mean ──────────────────────────────────────────────────
conus_t2 = conus['T2'].mean(dim='time')
lat2d_da = conus['lat']   # DataArray, dims (south_north, west_east)
lon2d_da = conus['lon']

# build a 2D boolean mask to identify cells within the bounding box
bounds_mask = (
    (lat2d_da >= LAT_MIN) & (lat2d_da <= LAT_MAX) &
    (lon2d_da >= LON_MIN) & (lon2d_da <= LON_MAX)
)
# find the rectangular row/col extents of the valid region
valid = bounds_mask.values
valid_rows = np.where(valid.any(axis=1))[0]
valid_cols = np.where(valid.any(axis=0))[0]
row_sl = slice(valid_rows[0], valid_rows[-1] + 1)
col_sl = slice(valid_cols[0], valid_cols[-1] + 1)

# use a plain rectangular slice for coordinates (no masked values in lat/lon)
lat2d = conus['lat'].values[row_sl, col_sl]
lon2d = conus['lon'].values[row_sl, col_sl]
# mask data values outside the bounding box with NaN
inner_mask = valid[row_sl, col_sl]
conus_arr = np.where(inner_mask, conus_t2.values[row_sl, col_sl], np.nan)

# ── shared colorscale (land-only values) ─────────────────────────────────────
vmin = min(np.nanmin(era_arr), np.nanmin(conus_arr))
vmax = max(np.nanmax(era_arr), np.nanmax(conus_arr))

# ── plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.05)

ax1 = fig.add_subplot(gs[0], projection=proj)
im1 = ax1.pcolormesh(lon_e, lat_e, era_arr, transform=ccrs.PlateCarree(),
                     cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
add_map_features(ax1, lat_side='left')
ax1.set_title('ERA5', fontsize=14, fontweight='bold', pad=10)

ax2 = fig.add_subplot(gs[1], projection=proj)
im2 = ax2.pcolormesh(lon2d, lat2d, conus_arr, transform=ccrs.PlateCarree(),
                     cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
add_map_features(ax2, lat_side='right')
ax2.set_title('CONUS404', fontsize=14, fontweight='bold', pad=10)

cax = fig.add_subplot(gs[2])
fig.colorbar(im2, cax=cax, extend='both').set_label('K', fontsize=12, fontweight='bold')

fig.suptitle(f'Yearly Mean 2m Temperature ({YEAR}) — Ocean Masked',
             fontsize=15, fontweight='bold', y=0.97)

out = "plots/t2m_heatmap_ocean_masked.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"saved: {out}")

era.close()
conus.close()
