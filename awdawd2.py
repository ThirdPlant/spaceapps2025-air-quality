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

import datatree  # correct import


############################################Connection:

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
# "Nitrogen Dioxide total column"
request = Request(
    collection=Collection(id="C2930725014-LARC_CLOUD"),
    granule_name=["TEMPO_NO2_L2_V03_20250406T215103Z_S012G07.nc"],
)
print(request.is_valid())

#######################################################Processing and download:

job_id = harmony_client.submit(request)
print(f"jobID = {job_id}")

harmony_client.wait_for_processing(job_id, show_progress=True)

results = harmony_client.download_all(job_id, directory="tmp")
all_results_stored = [f.result() for f in results]

print(f"Number of result files: {len(all_results_stored)}")


################################Open the data file:

# Open the data file using the Xarray package.
#   Alternatively, one could use the
#   netCDF4-python (https://unidata.github.io/netcdf4-python/) library.
datatree = xr.open_dataset(all_results_stored[0])
print(datatree)


