import xarray as xr

conus404 = xr.open_dataset("../../final_data/conus404_yearly_2015.nc")
era5 = xr.open_dataset("../../../sduan/pipeline/data/processed/era5_2015.nc")

print("---CONUS404 Data Vars---")
print(conus404.data_vars)
print("-------")
print("---ERA5 Data Vars---")
print(era5.data_vars)
