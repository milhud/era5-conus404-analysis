#!/usr/local/other/GEOSpyD/24.3.0-0/2024-08-29/envs/py3.12/bin/python3
"""
Lightweight exploration: find land/ocean mask proxy variables.
Uses lazy loading; only loads small slices into memory.
"""
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

ERA5_FILE  = "/discover/nobackup/hpmille1/data/era5_data/era5_2000.nc"
CONUS_FILE = "/discover/nobackup/hpmille1/data/conus404_data/conus404_yearly_2000.nc"

LAT_MIN, LAT_MAX = 24, 50
LON_MIN, LON_MAX = -125, -66

os.makedirs("land_mask_exploration", exist_ok=True)

proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39,
                             standard_parallels=(33, 45))

def make_map(ax):
    ax.coastlines('50m', color='black', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

era   = xr.open_dataset(ERA5_FILE)
conus = xr.open_dataset(CONUS_FILE)

# ── ERA5 lsm ─────────────────────────────────────────────────────────────────
print("\n=== ERA5: lsm (Land-sea mask) ===")
lsm_da = era['lsm'].isel(time=0, valid_time=0)
lsm_sub = lsm_da.sel(latitude=slice(LAT_MAX, LAT_MIN),
                      longitude=slice(LON_MIN, LON_MAX))
lsm_arr = lsm_sub.values
lat_e = lsm_sub['latitude'].values
lon_e = lsm_sub['longitude'].values
print(f"  lsm: min={lsm_arr.min():.3f}  max={lsm_arr.max():.3f}")
print(f"  fraction >= 0.5 (land): {(lsm_arr >= 0.5).mean()*100:.1f}%")
print(f"  fraction <  0.5 (ocean): {(lsm_arr < 0.5).mean()*100:.1f}%")

fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': proj})
im = ax.pcolormesh(lon_e, lat_e, lsm_arr, transform=ccrs.PlateCarree(),
                   cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
make_map(ax)
plt.colorbar(im, ax=ax, shrink=0.7, label='0=ocean, 1=land')
ax.set_title('ERA5 lsm – Land-sea mask', fontsize=12)
plt.savefig("land_mask_exploration/era5_lsm.png", dpi=100, bbox_inches='tight')
plt.close()
print("  saved: land_mask_exploration/era5_lsm.png")

# ── ERA5 z (geopotential as topography proxy) ────────────────────────────────
print("\n=== ERA5: z (Geopotential) ===")
z_da = era['z'].isel(time=0, valid_time=0)
z_sub = z_da.sel(latitude=slice(LAT_MAX, LAT_MIN),
                  longitude=slice(LON_MIN, LON_MAX))
z_arr = z_sub.values
print(f"  z: min={z_arr.min():.1f}  max={z_arr.max():.1f}  NaN%={np.isnan(z_arr).mean()*100:.1f}%")
# geopotential / g = geometric height (m)
z_m = z_arr / 9.80665
print(f"  z/g (m): min={z_m.min():.1f}  max={z_m.max():.1f}")

fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': proj})
im = ax.pcolormesh(lon_e, lat_e, z_m, transform=ccrs.PlateCarree(),
                   cmap='terrain', shading='auto')
make_map(ax)
plt.colorbar(im, ax=ax, shrink=0.7, label='m (z/g)')
ax.set_title('ERA5 z/g – Geopotential height (surface topography)', fontsize=12)
plt.savefig("land_mask_exploration/era5_z_height.png", dpi=100, bbox_inches='tight')
plt.close()
print("  saved: land_mask_exploration/era5_z_height.png")

# ── ERA5: t2m masked with lsm ────────────────────────────────────────────────
print("\n=== ERA5: t2m with lsm mask preview ===")
t2m_da = era['t2m']
time_dim = 'valid_time' if 'valid_time' in t2m_da.dims else 'time'
# use a single timestep to avoid memory issues
t2m_sub = t2m_da.isel(time=0).mean(dim='valid_time').sel(
    latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
t2m_arr = t2m_sub.values
t2m_masked = np.where(lsm_arr < 0.5, np.nan, t2m_arr)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': proj})
vmin, vmax = np.nanmin(t2m_arr), np.nanmax(t2m_arr)
im = axes[0].pcolormesh(lon_e, lat_e, t2m_arr, transform=ccrs.PlateCarree(),
                         cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
make_map(axes[0]); plt.colorbar(im, ax=axes[0], shrink=0.7, label='K')
axes[0].set_title('ERA5 t2m – unmasked', fontsize=12)

im2 = axes[1].pcolormesh(lon_e, lat_e, t2m_masked, transform=ccrs.PlateCarree(),
                          cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
make_map(axes[1]); plt.colorbar(im2, ax=axes[1], shrink=0.7, label='K')
axes[1].set_title('ERA5 t2m – ocean masked (lsm < 0.5 → NaN)', fontsize=12)
plt.suptitle('ERA5 2m Temp: unmasked vs lsm-masked (year 2000, annual mean)',
             fontsize=13, fontweight='bold')
plt.savefig("land_mask_exploration/era5_t2m_mask_preview.png", dpi=100, bbox_inches='tight')
plt.close()
print("  saved: land_mask_exploration/era5_t2m_mask_preview.png")

# ── CONUS404: check every variable for NaN over ocean region ─────────────────
print("\n=== CONUS404: NaN pattern check over Gulf of Mexico (ocean proxy) ===")
# Gulf of Mexico box in index space
lat2d = conus['lat'].values   # (south_north, west_east)
lon2d = conus['lon'].values
ocean_mask = ((lat2d >= 24) & (lat2d <= 30) & (lon2d >= -97) & (lon2d <= -82))
land_mask_approx = ((lat2d >= 30) & (lat2d <= 42) & (lon2d >= -100) & (lon2d <= -80))

print(f"  Ocean-proxy pixels in box: {ocean_mask.sum()}")
print(f"  Land-proxy pixels in box:  {land_mask_approx.sum()}")

for v in conus.data_vars:
    da = conus[v]
    if da.ndim < 2:
        continue
    if 'south_north' not in da.dims or 'west_east' not in da.dims:
        continue
    # take first time step only
    if 'time' in da.dims:
        da2d = da.isel(time=0)
    elif 'Time' in da.dims:
        da2d = da.isel(Time=0)
    else:
        da2d = da
    # take first level if 3D
    if da2d.ndim == 3:
        da2d = da2d.isel({[d for d in da2d.dims if d not in ('south_north','west_east')][0]: 0})
    arr = da2d.values
    if arr.shape != lat2d.shape:
        continue
    ocean_nan  = np.isnan(arr[ocean_mask]).mean()  if ocean_mask.sum() > 0 else np.nan
    land_nan   = np.isnan(arr[land_mask_approx]).mean() if land_mask_approx.sum() > 0 else np.nan
    global_nan = np.isnan(arr).mean()
    print(f"  {v:25s}  global_NaN%={global_nan*100:5.1f}  "
          f"ocean_NaN%={ocean_nan*100:5.1f}  land_NaN%={land_nan*100:5.1f}")

# ── CONUS404: plot ACRAINLSM and LAI to check if they are NaN over ocean ─────
print("\n=== Plotting CONUS404 ACRAINLSM and LAI ===")
for v in ['ACRAINLSM', 'LAI']:
    if v not in conus.data_vars:
        print(f"  {v} not found, skipping")
        continue
    da = conus[v]
    if 'time' in da.dims:
        arr = da.isel(time=0).values
    else:
        arr = da.values
    if arr.ndim == 3: arr = arr[0]
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': proj})
    im = ax.pcolormesh(lon2d, lat2d, arr, transform=ccrs.PlateCarree(),
                       cmap='YlGn', shading='auto')
    make_map(ax)
    plt.colorbar(im, ax=ax, shrink=0.7, label=conus[v].attrs.get('units',''))
    ax.set_title(f'CONUS404: {v}  —  {conus[v].attrs.get("long_name","")}', fontsize=11)
    out = f"land_mask_exploration/conus404_{v}.png"
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out}")

# ── CONUS404: T2 masked with ACRAINLSM NaN proxy ─────────────────────────────
print("\n=== CONUS404: T2 mask preview using ACRAINLSM NaN as land proxy ===")
t2_mean = conus['T2'].mean(dim='time').values

# build candidate masks
masks = {}

if 'ACRAINLSM' in conus.data_vars:
    acrainlsm_t0 = conus['ACRAINLSM'].isel(time=0).values
    if acrainlsm_t0.ndim == 3: acrainlsm_t0 = acrainlsm_t0[0]
    masks['ACRAINLSM_nan_proxy'] = ~np.isnan(acrainlsm_t0)  # True = land (not NaN)

if 'LAI' in conus.data_vars:
    lai_t0 = conus['LAI'].isel(time=0).values
    if lai_t0.ndim == 3: lai_t0 = lai_t0[0]
    masks['LAI_nan_proxy']       = ~np.isnan(lai_t0)

# Also try: mask where annual-mean T2 itself shows suspicious constant values
# (WRF sea cells often have T2 = SST which looks like smooth ocean signal)

n_masks = len(masks)
if n_masks == 0:
    print("  No mask candidates found for CONUS404.")
else:
    fig, axes = plt.subplots(1, n_masks + 1, figsize=(8*(n_masks+1), 6),
                              subplot_kw={'projection': proj})
    if n_masks == 1:
        axes = [axes[0], axes[1]] if hasattr(axes, '__len__') else axes
    vmin_c, vmax_c = np.nanmin(t2_mean), np.nanmax(t2_mean)

    im = axes[0].pcolormesh(lon2d, lat2d, t2_mean, transform=ccrs.PlateCarree(),
                             cmap='RdYlBu_r', vmin=vmin_c, vmax=vmax_c, shading='auto')
    make_map(axes[0]); plt.colorbar(im, ax=axes[0], shrink=0.7, label='K')
    axes[0].set_title('CONUS404 T2 – unmasked', fontsize=12)

    for i, (mname, land_bool) in enumerate(masks.items()):
        t2_m = np.where(land_bool, t2_mean, np.nan)
        im2 = axes[i+1].pcolormesh(lon2d, lat2d, t2_m, transform=ccrs.PlateCarree(),
                                    cmap='RdYlBu_r', vmin=vmin_c, vmax=vmax_c, shading='auto')
        make_map(axes[i+1]); plt.colorbar(im2, ax=axes[i+1], shrink=0.7, label='K')
        axes[i+1].set_title(f'T2 masked ({mname})', fontsize=11)

    plt.suptitle('CONUS404 T2: mask candidates (year 2000 annual mean)',
                 fontsize=13, fontweight='bold')
    out = "land_mask_exploration/conus404_t2_mask_preview.png"
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out}")

era.close()
conus.close()
print("\nDone. Check land_mask_exploration/")
