import datetime as dt
import getpass

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from xarray.plot.utils import label_from_attrs

from harmony import BBox, Client, Collection, Request
from harmony.config import Environment

import tempfile

import os
from dotenv import load_dotenv, dotenv_values

import netCDF4 as nc




load_dotenv()


print("Using Earthdata Login credentials from environment variables")
username = os.getenv("EARTHDATA_USERNAME")
password = os.getenv("EARTHDATA_PASSWORD")

if not username or not password:
    raise ValueError("Earthdata credentials are not set in environment variables")

harmony_client = Client(env=Environment.PROD, auth=(username, password))

# "Nitrogen Dioxide total column"
request = Request(
    collection=Collection(id="C2930725014-LARC_CLOUD"),
    granule_name=["TEMPO_NO2_L2_V03_20250406T215103Z_S012G07.nc"],
)
request.is_valid()

job_id = harmony_client.submit(request)
print(f"jobID = {job_id}")

harmony_client.wait_for_processing(job_id, show_progress=True)

# Use a temporary directory for storing results within the project structure

project_tmp_dir = os.path.join(os.getcwd(), "tmp")
os.makedirs(project_tmp_dir, exist_ok=True)

results = harmony_client.download_all(job_id, directory=project_tmp_dir)
all_results_stored = [f.result() for f in results]

print(f"Number of result files: {len(all_results_stored)}")

datatree = xr.open_dataset(all_results_stored[0], engine="netcdf4")
datatree

# Method 1: Use netCDF4 to explore the full structure
print("=== Exploring TEMPO NO2 file structure ===")
with nc.Dataset(all_results_stored[0], 'r') as ncfile:
    print(f"Root groups: {list(ncfile.groups.keys())}")
    
    # Check each group for NO2 variables
    for group_name in ncfile.groups.keys():
        group = ncfile.groups[group_name]
        print(f"\nGroup '{group_name}' variables:")
        for var_name in group.variables.keys():
            var = group.variables[var_name]
            long_name = getattr(var, 'long_name', 'No description')
            print(f"  - {var_name}: {long_name}")

# Method 2: Open specific groups with xarray
print("\n=== Loading NO2 data from product group ===")
try:
    # TEMPO files typically have data in 'product' group
    product_ds = xr.open_dataset(all_results_stored[0], engine="netcdf4", group='product')
    print(f"Product group variables: {list(product_ds.data_vars.keys())}")
    
    # Look for tropospheric NO2 column
    no2_vars = [var for var in product_ds.data_vars.keys() 
                if any(keyword in var.lower() for keyword in ['vertical_column', 'no2', 'troposphere'])]
    
    if no2_vars:
        var_name = no2_vars[0]  # Use first match
        no2_data = product_ds[var_name]
        print(f"\nFound NO2 variable: {var_name}")
        print(f"Shape: {no2_data.shape}")
        print(f"Units: {no2_data.attrs.get('units', 'unknown')}")
        print(f"Description: {no2_data.attrs.get('long_name', 'No description')}")
        print(no2_data)
        
        # Store for further analysis
        da = no2_data
        
    else:
        # If no obvious NO2 variables, show all available
        print("\nNo obvious NO2 variables found. Available variables:")
        for var in product_ds.data_vars:
            attrs = product_ds[var].attrs
            print(f"  - {var}: {attrs.get('long_name', attrs.get('description', 'No description'))}")
            
except Exception as e:
    print(f"Error accessing product group: {e}")
    
    # Fallback: try other common group names
    common_groups = ['geolocation', 'support_data', 'meteorology']
    for group_name in common_groups:
        try:
            group_ds = xr.open_dataset(all_results_stored[0], engine="netcdf4", group=group_name)
            print(f"\n{group_name} group variables: {list(group_ds.data_vars.keys())}")
        except:
            continue

product_variable_name = "vertical_column_troposphere"
da = product_ds[product_variable_name]
da

data_proj = ccrs.PlateCarree()


def make_nice_map(axis):
    axis.add_feature(cfeature.STATES, color="gray", lw=0.1)
    axis.coastlines(resolution="50m", color="gray", linewidth=0.5)

    axis.set_extent([-150, -40, 14, 65], crs=data_proj)
    grid = axis.gridlines(draw_labels=["left", "bottom"], dms=True)
    grid.xformatter = LONGITUDE_FORMATTER
    grid.yformatter = LATITUDE_FORMATTER

    # Load geolocation data from the geolocation group
geolocation_ds = xr.open_dataset(all_results_stored[0], engine="netcdf4", group='geolocation')
print(f"Geolocation group variables: {list(geolocation_ds.data_vars.keys())}")

# Now use the geolocation data for plotting
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": data_proj})

make_nice_map(ax)

contour_handle = ax.contourf(
    geolocation_ds["longitude"],
    geolocation_ds["latitude"], 
    da.where(product_ds["main_data_quality_flag"] == 0),
    levels=100,
    vmin=0,
    zorder=2,
)

cb = plt.colorbar(contour_handle)
cb.set_label(label_from_attrs(da))

plt.show()