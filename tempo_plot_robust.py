#!/usr/bin/env python3
"""
Robust TEMPO NASA Data Visualization
A version that handles common Windows path issues and provides detailed error messages.
"""

import os
import sys
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from harmony import Client, Collection, Request
from harmony.config import Environment

def check_dependencies():
    """Check if all required packages are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('numpy', 'np'),
        ('xarray', 'xr'),
        ('matplotlib', 'plt'),
        ('cartopy', 'ccrs'),
        ('harmony', 'Client')
    ]
    
    missing = []
    for package, alias in required_packages:
        try:
            if package == 'harmony':
                from harmony import Client
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Please install them with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies available!")
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    dirs_to_create = ['tempo_data', 'output']
    
    for dir_name in dirs_to_create:
        try:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ Created/verified directory: {dir_name}")
        except Exception as e:
            print(f"❌ Error creating directory {dir_name}: {e}")
            return False
    
    return True

def get_absolute_path(relative_path):
    """Convert relative path to absolute path."""
    return os.path.abspath(relative_path)

def main():
    """Main function with robust error handling."""
    
    print("🌍 TEMPO NASA Data Visualization (Robust Version)")
    print("=" * 55)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return
    
    # Create directories
    if not create_directories():
        print("\n❌ Failed to create necessary directories.")
        return
    
    # Your Earthdata credentials
    username = "linas03"
    password = "Mumis.Jurenas.789"
    
    try:
        # 1. Connect to Harmony
        print("\n📡 Connecting to NASA Harmony...")
        print(f"   Username: {username}")
        
        harmony_client = Client(env=Environment.PROD, auth=(username, password))
        print("✅ Connected successfully!")
        
        # 2. Request specific TEMPO data
        print("\n🔍 Requesting TEMPO NO2 data...")
        request = Request(
            collection=Collection(id="C2930725014-LARC_CLOUD"),
            granule_name=["TEMPO_NO2_L2_V03_20250406T215103Z_S012G07.nc"]
        )
        
        print(f"   Request valid: {request.is_valid()}")
        
        # 3. Submit and download
        print("\n📥 Submitting request...")
        job_id = harmony_client.submit(request)
        print(f"   Job ID: {job_id}")
        
        print("\n⏳ Processing...")
        harmony_client.wait_for_processing(job_id, show_progress=True)
        
        print("\n💾 Downloading data...")
        download_dir = get_absolute_path("tempo_data")
        print(f"   Download directory: {download_dir}")
        
        results = harmony_client.download_all(job_id, directory=download_dir)
        files = [f.result() for f in results]
        print(f"✅ Downloaded {len(files)} file(s)")
        
        if not files:
            print("❌ No files downloaded!")
            return
        
        # 4. Load the data
        print("\n📊 Loading data...")
        file_path = get_absolute_path(files[0])
        print(f"   File path: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        # Load product data (NO2 measurements)
        print("   Loading product data...")
        product_ds = xr.open_dataset(file_path, engine="netcdf4", group='product')
        
        # Load geolocation data (lat/lon coordinates)
        print("   Loading geolocation data...")
        geo_ds = xr.open_dataset(file_path, engine="netcdf4", group='geolocation')
        
        # Get NO2 tropospheric column data
        no2_data = product_ds['vertical_column_troposphere']
        
        print(f"📈 Data shape: {no2_data.shape}")
        print(f"📏 Units: {no2_data.attrs.get('units', 'unknown')}")
        print(f"📝 Description: {no2_data.attrs.get('long_name', 'NO2 data')}")
        
        # 5. Create visualization
        print("\n🎨 Creating visualization...")
        
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
        print("   Applying quality filter...")
        quality_mask = product_ds['main_data_quality_flag'] == 0
        data_to_plot = no2_data.where(quality_mask)
        
        # Create the plot
        print("   Creating contour plot...")
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
        output_file = get_absolute_path("tempo_no2_map_robust.png")
        print(f"   Saving plot to: {output_file}")
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved as: {output_file}")
        
        # Show the plot
        print("   Displaying plot...")
        plt.show()
        
        print("\n🎉 Success! TEMPO data visualization complete!")
        print(f"📁 Data files saved in: {download_dir}")
        print(f"🖼️  Visualization saved as: {output_file}")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found error: {e}")
        print("This might be a path issue. Try running from the correct directory.")
    except PermissionError as e:
        print(f"\n❌ Permission error: {e}")
        print("Try running as administrator or check file permissions.")
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please install missing packages with: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
