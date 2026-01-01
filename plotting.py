import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ks_2samp, linregress
from pathlib import Path

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