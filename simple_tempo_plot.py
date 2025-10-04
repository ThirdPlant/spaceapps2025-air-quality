#!/usr/bin/env python3
"""
Simple TEMPO NASA Data Visualization
A streamlined script to access and plot TEMPO NO2 data using Harmony.
"""

import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from harmony import Client, Collection, Request
from harmony.config import Environment

def main():
    """Main function to access and visualize TEMPO data."""
    
    # Your Earthdata credentials
    username = "linas03"
    password = "Mumis.Jurenas.789"
    
    print("🌍 Accessing TEMPO NASA Data...")
    print("=" * 40)
    
    try:
        # 1. Connect to Harmony
        print("📡 Connecting to NASA Harmony...")
        harmony_client = Client(env=Environment.PROD, auth=(username, password))
        print("✅ Connected successfully!")
        
        # 2. Request specific TEMPO data
        print("🔍 Requesting TEMPO NO2 data...")
        request = Request(
            collection=Collection(id="C2930725014-LARC_CLOUD"),
            granule_name=["TEMPO_NO2_L2_V03_20250406T215103Z_S012G07.nc"]
        )
        
        # 3. Submit and download
        print("📥 Submitting request...")
        job_id = harmony_client.submit(request)
        print(f"Job ID: {job_id}")
        
        print("⏳ Processing...")
        harmony_client.wait_for_processing(job_id, show_progress=True)
        
        print("💾 Downloading data...")
        os.makedirs("tempo_data", exist_ok=True)
        results = harmony_client.download_all(job_id, directory="tempo_data")
        files = [f.result() for f in results]
        print(f"✅ Downloaded {len(files)} file(s)")
        
        # 4. Load the data
        print("📊 Loading data...")
        file_path = files[0]
        
        # Load product data (NO2 measurements)
        product_ds = xr.open_dataset(file_path, engine="netcdf4", group='product')
        
        # Load geolocation data (lat/lon coordinates)
        geo_ds = xr.open_dataset(file_path, engine="netcdf4", group='geolocation')
        
        # Get NO2 tropospheric column data
        no2_data = product_ds['vertical_column_troposphere']
        
        print(f"📈 Data shape: {no2_data.shape}")
        print(f"📏 Units: {no2_data.attrs.get('units', 'unknown')}")
        print(f"📝 Description: {no2_data.attrs.get('long_name', 'NO2 data')}")
        
        # 5. Create visualization
        print("🎨 Creating visualization...")
        
        # Set up the map
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, color="gray", linewidth=0.5)
        ax.add_feature(cfeature.STATES, color="gray", linewidth=0.3)
        
        # Set map extent (North America)
        ax.set_extent([-150, -40, 14, 65], crs=ccrs.PlateCarree())
        
        # Add gridlines
        grid = ax.gridlines(draw_labels=["left", "bottom"], dms=True, alpha=0.5)
        grid.xformatter = LONGITUDE_FORMATTER
        grid.yformatter = LATITUDE_FORMATTER
        
        # Apply quality filter (only show good quality data)
        quality_mask = product_ds['main_data_quality_flag'] == 0
        data_to_plot = no2_data.where(quality_mask)
        
        # Create the plot
        contour = ax.contourf(
            geo_ds["longitude"],
            geo_ds["latitude"], 
            data_to_plot,
            levels=50,
            vmin=0,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            zorder=2
        )
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(f"NO2 Tropospheric Column [{no2_data.attrs.get('units', '')}]")
        
        # Add title
        plt.title("TEMPO NO2 Tropospheric Column\nApril 6, 2025", fontsize=14, pad=20)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = "tempo_no2_map.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved as: {output_file}")
        
        # Show the plot
        plt.show()
        
        print("\n🎉 Success! TEMPO data visualization complete!")
        print(f"📁 Data files saved in: tempo_data/")
        print(f"🖼️  Visualization saved as: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
