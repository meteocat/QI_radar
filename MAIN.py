from Polar2Cartesian_PPI import Polar2Cartesian
from CAPPI_LUE_tools import make_CAPPI, make_LUE
from Composite_tools import composite
from FindIRISFiles import search_short_range, search_long_range
from Import_config import load_config

import os
import numpy as np
import xarray as xr
import datetime as dt
from time import time
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

def distance_weighting(dist):
    ''' Weighting function based on distance quality index (QH)
    
    Input:
    -----
    dist : float or 2D array
        Distance value(s) in meters to be used for weighting. Can be a single
        numeric value or a 2D array of distances.

    Output:
    ------
    QH : float or 2D array
        Weighting value(s) based on distance, with the same shape as the input.
        Values corresponding to negative distances are set to 0.
    '''

    # Define scale height (in meters)
    H = 1000

    # Compute quality index based on distance
    QH = (np.exp(-dist**2/H**2)) ** (1/3)
    QH[dist < 0] = 0 # Set weighting to 0 for negative distances
    
    return QH

def save_dataset(Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP, x, y, 
                 filedate, prod_type, comp_type, product_save_dir, VOLUME):
    '''
    Results datasets saving function. If necessary, creates all the directories.
    
    :param Z_COMP: Array of composite reflectivity values
    :param QI_COMP: Array of composite quality index values
    :param RAD_COMP: Array of composite radar selection values
    :param ELEV_COMP: Array of composite elevation beam values
    :param x: Array of x coordinates
    :param y: Array of y coordinates
    :param filedate: File date string in 'yymmddHHMM' format
    :param prod_type: Product type string, e.g., 'CAPPI1.5km' or 'LUE'
    :param comp_type: Composite type string, e.g., 'MAXZ' or 'MAXQI'
    :param product_save_dir: Directory path to save the product
    :param VOLUME: Volume type string, e.g., 'VOLB', 'VOLA', or 'VOLBC'
    '''

    # Create xarray dataset
    result = xr.Dataset({"Z": (["y", "x"], Z_COMP), 
                        "QI": (["y", "x"], QI_COMP), 
                        "RAD": (["y", "x"], RAD_COMP),
                        "ELEV": (["y", "x"], ELEV_COMP)},
                        coords={"x": x, "y": y})
    
    # Create all necessary directories
    os.makedirs(f"{product_save_dir}", exist_ok=True)
    os.makedirs(f"{product_save_dir}/{VOLUME}", exist_ok=True)
    os.makedirs(f"{product_save_dir}/{VOLUME}/{prod_type}", exist_ok=True)
    todate_dir = f"{product_save_dir}/{VOLUME}/{prod_type}/{comp_type}"
    os.makedirs(todate_dir, exist_ok=True)
    time_dt = dt.datetime.strptime(filedate, '%y%m%d%H%M')
    yy, mm, dd = time_dt.strftime("%Y"), time_dt.strftime("%m"), time_dt.strftime("%d")
    os.makedirs(f"{todate_dir}/{yy}", exist_ok=True)
    os.makedirs(f"{todate_dir}/{yy}/{mm}", exist_ok=True)
    save_dir = f"{todate_dir}/{yy}/{mm}/{dd}"
    os.makedirs(save_dir, exist_ok=True)

    # Define filename and save dataset
    filename = f"{VOLUME}_{prod_type}_{comp_type}_{filedate}.nc"
    save_as = f"{save_dir}/{filename}"
    result.to_netcdf(save_as, engine="scipy")
    print(f"Created {filename}")

# Load configuration parameters from "config" file
config = load_config("config.txt")

init_dt = config["init_dt"]
fin_dt = config["fin_dt"]
VOLUME = config["VOLUME"]
COMP_types = config["COMP_types"]
CAPPI_H = config["CAPPI_H"]
dl = config["dl"]

PPI_save_dir = config["PPI_save_dir"]
product_save_dir = config["product_save_dir"]
IRIS_dir = config["IRIS_dir"]
TOP12_clim_path = config["TOP12_clim_path"]
DEM_path = config["SR_DEM_path"] if VOLUME != 'VOLA' else config["LR_DEM_path"]

# Import DEM data
with rasterio.open(DEM_path) as src:
    DEM_values = src.read(1)
    height, width = src.shape
    transform = src.transform
x = np.arange(width) * transform.a + transform.c
y = np.arange(height) * transform.e + transform.f
DEM_coords = np.array(np.meshgrid(x, y))
DEM_coords = np.moveaxis(DEM_coords, 0, 2)

# Loop over time range with 6-minute intervals
for dt_time in np.arange(init_dt, fin_dt, dt.timedelta(minutes=6)):
    # Convert numpy datetime64 to datetime.datetime and extract time components
    dt_time = dt_time.astype(dt.datetime)
    yy, mm, dd, hh, MM = dt_time.year, dt_time.month, dt_time.day, dt_time.hour, dt_time.minute
    IRIS_time = (yy, mm, dd, hh, MM)

    # Clear folder where temporal PPIs will be saved
    for filename in os.listdir(PPI_save_dir):
        file_path = os.path.join(PPI_save_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    # Search IRIS file paths
    if VOLUME != 'VOLA':
        paths = search_short_range(IRIS_time, IRIS_dir)
    else:
        paths = search_long_range(IRIS_time, IRIS_dir)

    # Define initial time of execution
    t0 = time()
        
    # ============================= INDIVIDUAL PPI COMPUTATION =============================

    # Loop over radar files
    i = 0
    for n in range(0, len(paths), 2 if VOLUME != "VOLA" else 1):
        try:
            # Transform polar to cartesian coordinates for each PPI according 
            # to the volume type used
            if VOLUME == 'VOLA' or VOLUME == 'VOLB':
                ds = Polar2Cartesian(paths[n], TOP12_clim_path, 
                                        DEM_values, DEM_coords, 
                                        dl=dl, save_dir=PPI_save_dir)
            elif VOLUME == 'VOLBC':
                ds_VOLB = Polar2Cartesian(paths[n], TOP12_clim_path, 
                                            DEM_values, DEM_coords, 
                                            dl=dl, save_dir=PPI_save_dir)
                ds_VOLC = Polar2Cartesian(paths[n+1], TOP12_clim_path, 
                                            DEM_values, DEM_coords, 
                                            dl=dl, save_dir=PPI_save_dir)
                ds = xr.concat([ds_VOLB, ds_VOLC], dim="elev")

            # Create temorary array for storing each radar individual products
            if i==0:
                CAPPI_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan     # Single-radar CAPPI reflectivity
                QICAPPI_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan   # Single-radar CAPPI QI
                ELEVCAPPI_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan # Single-radar CAPPI ELEV
                LUE_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan       # Single-radar LUE reflectivity
                QILUE_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan     # Single-radar LUE QI
                ELEVLUE_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan   # Single-radar LUE ELEV

                # Resample DEM to match radar grid
                xgrid, ygrid = ds.x.values, ds.y.values
                x_min, x_max = xgrid.min(), xgrid.max()
                y_min, y_max = ygrid.min(), ygrid.max()
                new_transform = from_origin(x_min, y_max, dl, dl)
                dst_shape = (len(ygrid), len(xgrid))
                DEM_resampled = np.empty(dst_shape, dtype=np.float32)
                reproject(
                    source=DEM_values,
                    destination=DEM_resampled,
                    src_transform=transform,
                    src_crs="EPSG:4326",
                    dst_transform=new_transform,
                    dst_crs="EPSG:25831",
                    resampling=Resampling.nearest
                )
            
            # Apply height-to-CAPPI quality index
            ds_CAPPI = ds.copy(deep=True)
            for e in range(len(ds.elev.values)):
                ds_e = ds_CAPPI.isel(elev=e)
                Z_e = ds_e.Z.values
                QI_e = ds_e.QI.values
                H_to_CAPPI = np.abs(ds_e.H.values - CAPPI_H)
                QI_e[Z_e != -32] = QI_e[Z_e != -32] * distance_weighting(H_to_CAPPI)[Z_e != -32]
                ds_CAPPI["QI"].values[e, ...] = QI_e
            
            # Compute and store single-radar CAPPI products
            CAPPI, QI, ELEV = make_CAPPI(ds_CAPPI, CAPPI_H)
            CAPPI_ind_rad[i, ...] = CAPPI
            QICAPPI_ind_rad[i, ...] = QI
            ELEVCAPPI_ind_rad[i, ...] = ELEV

            # Apply height-to-ground quality index
            ds_LUE = ds.copy(deep=True)
            for e in range(len(ds.elev.values)):
                ds_e = ds_LUE.isel(elev=e)
                Z_e = ds_e.Z.values
                QI_e = ds_e.QI.values
                H_to_ground = ds_e.H.values - DEM_resampled
                QI_e[Z_e != -32] = QI_e[Z_e != -32] * distance_weighting(H_to_ground)[Z_e != -32]
                ds_LUE["QI"].values[e, ...] = QI_e
            
            # Compute and store single-radar LUE products
            LUE, QI, H, ELEV = make_LUE(ds_LUE, DEM_resampled)
            LUE_ind_rad[i, ...] = LUE
            QILUE_ind_rad[i, ...] = QI
            ELEVLUE_ind_rad[i, ...] = ELEV
        
        except Exception as e:
            print(f"\nNot able to compute {paths[n]}\n{e}\n")

        i += 1

    # Iterate over composite types
    for comp_type in COMP_types:
        filedate = dt_time.strftime('%y%m%d%H%M') # File date string

        # =================================== CAPPI COMPOSITES ===================================

        # Compute CAPPI composite
        Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP = composite(CAPPI_ind_rad, QICAPPI_ind_rad, 
                                               ELEVCAPPI_ind_rad, comp_type)

        # Save results into a dataset
        prod_type = f'CAPPI{CAPPI_H/1000}km'
        save_dataset(Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP, ds.x.values, ds.y.values, 
                     filedate, prod_type, comp_type, product_save_dir, VOLUME)

        # ==================================== LUE COMPOSITES ====================================

        # Compute LUE composite
        Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP = composite(LUE_ind_rad, QILUE_ind_rad, 
                                               ELEVLUE_ind_rad, comp_type)

        # Save results into a dataset
        prod_type = f'LUE'
        save_dataset(Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP, ds.x.values, ds.y.values, 
                     filedate, prod_type, comp_type, product_save_dir, VOLUME)

    # End and print time of execution
    t1 = time()
    T = t1 - t0
    print(f"Compilation time: {int(T/60)}m{int(T%60)}s")

    print()