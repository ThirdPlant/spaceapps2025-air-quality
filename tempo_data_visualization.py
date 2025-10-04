#!/usr/bin/env python3
"""
TEMPO NASA Data Access and Visualization Script
This script accesses TEMPO (Tropospheric Emissions: Monitoring of Pollution) data
using NASA's Harmony service and creates visualizations with matplotlib.
"""

import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from xarray.plot.utils import label_from_attrs
import netCDF4 as nc
from datetime import datetime, timedelta

from harmony import BBox, Client, Collection, Request
from harmony.config import Environment

def setup_harmony_client(username, password):
    """Set up Harmony client with Earthdata credentials."""
    print(f"Setting up Harmony client for user: {username}")
    try:
        harmony_client = Client(env=Environment.PROD, auth=(username, password))
        print("✓ Successfully connected to Harmony")
        return harmony_client
    except Exception as e:
        print(f"✗ Error connecting to Harmony: {e}")
        raise

def search_tempo_data(harmony_client, start_date=None, end_date=None, bbox=None, use_specific_granule=False):
    """Search for available TEMPO data."""
    # TEMPO NO2 collection ID
    collection_id = "C2930725014-LARC_CLOUD"
    
    if use_specific_granule:
        # Use a specific granule that we know exists (from your original code)
        print("Using specific TEMPO granule: TEMPO_NO2_L2_V03_20250406T215103Z_S012G07.nc")
        request = Request(
            collection=Collection(id=collection_id),
            granule_name=["TEMPO_NO2_L2_V03_20250406T215103Z_S012G07.nc"]
        )
    else:
        if start_date is None:
            start_date = datetime(2024, 1, 1)  # Go back further to find data
        if end_date is None:
            end_date = datetime(2024, 12, 31)
        
        print(f"Searching for TEMPO data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Create request
        request = Request(
            collection=Collection(id=collection_id),
            temporal={
                "start": start_date,
                "stop": end_date
            }
        )
        
        if bbox:
            request.spatial = BBox(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3])
    
    print(f"Request valid: {request.is_valid()}")
    return request

def download_tempo_data(harmony_client, request, output_dir="tempo_data"):
    """Download TEMPO data using Harmony."""
    print("Submitting request to Harmony...")
    job_id = harmony_client.submit(request)
    print(f"Job ID: {job_id}")
    
    print("Waiting for processing...")
    harmony_client.wait_for_processing(job_id, show_progress=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading results...")
    results = harmony_client.download_all(job_id, directory=output_dir)
    all_results = [f.result() for f in results]
    
    print(f"Downloaded {len(all_results)} files")
    return all_results

def explore_tempo_file(file_path):
    """Explore the structure of a TEMPO NetCDF file."""
    print(f"\n=== Exploring TEMPO file: {os.path.basename(file_path)} ===")
    
    with nc.Dataset(file_path, 'r') as ncfile:
        print(f"File format: {ncfile.data_model}")
        print(f"Root groups: {list(ncfile.groups.keys())}")
        
        # Explore each group
        for group_name in ncfile.groups.keys():
            group = ncfile.groups[group_name]
            print(f"\n--- Group '{group_name}' ---")
            print(f"Variables: {list(group.variables.keys())}")
            
            # Show variable details
            for var_name in group.variables.keys():
                var = group.variables[var_name]
                long_name = getattr(var, 'long_name', 'No description')
                units = getattr(var, 'units', 'No units')
                shape = var.shape
                print(f"  {var_name}: {long_name} [{units}] {shape}")

def load_tempo_data(file_path):
    """Load TEMPO data from NetCDF file."""
    print(f"\n=== Loading TEMPO data ===")
    
    # Load product data (contains the main measurements)
    try:
        product_ds = xr.open_dataset(file_path, engine="netcdf4", group='product')
        print(f"Product group loaded successfully")
        print(f"Variables: {list(product_ds.data_vars.keys())}")
    except Exception as e:
        print(f"Error loading product group: {e}")
        return None, None, None
    
    # Load geolocation data
    try:
        geo_ds = xr.open_dataset(file_path, engine="netcdf4", group='geolocation')
        print(f"Geolocation group loaded successfully")
        print(f"Variables: {list(geo_ds.data_vars.keys())}")
    except Exception as e:
        print(f"Error loading geolocation group: {e}")
        return product_ds, None, None
    
    # Find NO2 data variable
    no2_vars = [var for var in product_ds.data_vars.keys() 
                if any(keyword in var.lower() for keyword in ['vertical_column', 'no2', 'troposphere'])]
    
    if not no2_vars:
        print("No NO2 variables found. Available variables:")
        for var in product_ds.data_vars:
            attrs = product_ds[var].attrs
            print(f"  - {var}: {attrs.get('long_name', attrs.get('description', 'No description'))}")
        return product_ds, geo_ds, None
    
    no2_var = no2_vars[0]
    no2_data = product_ds[no2_var]
    
    print(f"Selected NO2 variable: {no2_var}")
    print(f"Shape: {no2_data.shape}")
    print(f"Units: {no2_data.attrs.get('units', 'unknown')}")
    print(f"Description: {no2_data.attrs.get('long_name', 'No description')}")
    
    return product_ds, geo_ds, no2_data

def create_visualization(product_ds, geo_ds, no2_data, output_file=None):
    """Create a visualization of the TEMPO data."""
    print("\n=== Creating visualization ===")
    
    # Set up the map projection
    data_proj = ccrs.PlateCarree()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": data_proj})
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, color="gray", linewidth=0.5)
    ax.add_feature(cfeature.STATES, color="gray", linewidth=0.3)
    
    # Set extent (North America)
    ax.set_extent([-150, -40, 14, 65], crs=data_proj)
    
    # Add gridlines
    grid = ax.gridlines(draw_labels=["left", "bottom"], dms=True, alpha=0.5)
    grid.xformatter = LONGITUDE_FORMATTER
    grid.yformatter = LATITUDE_FORMATTER
    
    # Apply quality filter if available
    if 'main_data_quality_flag' in product_ds.data_vars:
        quality_mask = product_ds['main_data_quality_flag'] == 0
        data_to_plot = no2_data.where(quality_mask)
        print("Applied quality filter")
    else:
        data_to_plot = no2_data
        print("No quality filter available")
    
    # Create contour plot
    try:
        contour = ax.contourf(
            geo_ds["longitude"],
            geo_ds["latitude"], 
            data_to_plot,
            levels=50,
            vmin=0,
            transform=data_proj,
            cmap='viridis',
            zorder=2
        )
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(f"{no2_data.attrs.get('long_name', 'NO2')} [{no2_data.attrs.get('units', '')}]")
        
        # Add title
        plt.title(f"TEMPO {no2_data.attrs.get('long_name', 'NO2 Data')}\n"
                 f"Units: {no2_data.attrs.get('units', 'Unknown')}", 
                 fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print("This might be due to coordinate mismatch or data format issues")

def main():
    """Main function to run the TEMPO data access and visualization."""
    # User credentials
    username = "linas03"
    password = "Mumis.Jurenas.789"
    
    print("TEMPO NASA Data Access and Visualization")
    print("=" * 50)
    
    try:
        # Set up Harmony client
        harmony_client = setup_harmony_client(username, password)
        
        # Search for TEMPO data (try specific granule first)
        request = search_tempo_data(harmony_client, use_specific_granule=True)
        
        # Download data
        files = download_tempo_data(harmony_client, request)
        
        if not files:
            print("No files downloaded!")
            return
        
        # Process the first file
        file_path = files[0]
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        
        # Explore file structure
        explore_tempo_file(file_path)
        
        # Load data
        product_ds, geo_ds, no2_data = load_tempo_data(file_path)
        
        if no2_data is not None:
            # Create visualization
            create_visualization(product_ds, geo_ds, no2_data, "tempo_no2_visualization.png")
        else:
            print("Could not load NO2 data for visualization")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
