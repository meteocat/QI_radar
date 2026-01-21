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

def save_dataset(Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP, x, y, 
                 filedate, prod_type, comp_type, product_save_dir, VOLUME):
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

    filename = f"{VOLUME}_{prod_type}_{comp_type}_{filedate}.nc"
    save_as = f"{save_dir}/{filename}"

    result.to_netcdf(save_as, engine="scipy")
    print(f"Created {filename}")

config = load_config("config.txt")

init_dt = config["init_dt"]
fin_dt = config["fin_dt"]
VOLUME = config["VOLUME"]
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

for dt_time in np.arange(init_dt, fin_dt, dt.timedelta(minutes=6)):
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

    i = 0
    for n in range(0, len(paths), 2 if VOLUME != "VOLA" else 1):
        try:
            # Compute PPI data (only VOL-B if height <= 2000)
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
                CAPPI_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan
                QICAPPI_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan
                ELEVCAPPI_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan
                LUE_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan
                QILUE_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan
                ELEVLUE_ind_rad = np.ones((4, len(ds.y), len(ds.x))) * np.nan

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
            
            # Compute and store individual CAPPI (and QI)
            CAPPI, QI, ELEV = make_CAPPI(ds, CAPPI_H)
            CAPPI_ind_rad[i, ...] = CAPPI
            QICAPPI_ind_rad[i, ...] = QI
            ELEVCAPPI_ind_rad[i, ...] = ELEV

            # Compute and store individual close to LUEain products
            LUE, QI, H, ELEV = make_LUE(ds, DEM_resampled)
            LUE_ind_rad[i, ...] = LUE
            QILUE_ind_rad[i, ...] = QI
            ELEVLUE_ind_rad[i, ...] = ELEV
        
        except Exception as e:
            print(f"\nNot able to compute {paths[n]}\n{e}\n")

        i += 1


    for comp_type in ["MAXZ", "MAXQI"]:
        filedate = os.path.splitext(os.path.basename(paths[0]))[0][3:-2]

        # =================================== CAPPI COMPOSITES ===================================

        Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP = composite(CAPPI_ind_rad, QICAPPI_ind_rad, 
                                               ELEVCAPPI_ind_rad, comp_type)

        # Save results into a dataset
        prod_type = f'CAPPI{CAPPI_H/1000}km'
        save_dataset(Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP, ds.x.values, ds.y.values, 
                     filedate, prod_type, comp_type, product_save_dir, VOLUME)

        # ============================= CLOSE TO LUEAIN COMPOSITES =============================

        Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP = composite(LUE_ind_rad, QILUE_ind_rad, 
                                               ELEVLUE_ind_rad, comp_type)

        # Save results into a dataset
        prod_type = f'LUE'
        save_dataset(Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP, ds.x.values, ds.y.values, 
                     filedate, prod_type, comp_type, product_save_dir, VOLUME)

    # End and store time of execution
    t1 = time()
    T = t1 - t0
    print(f"Compilation time: {int(T/60)}m{int(T%60)}s")

    print()