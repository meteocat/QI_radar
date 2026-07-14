from polar2cartesian_tools import polar2cartesian
from CAPPI_LUE_tools import make_CAPPI, make_LUE

from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import datetime as dt
import argparse
import os
import logging
import time
import numpy as np
import xarray as xr
import rasterio

# Read command-line arguments
parser = argparse.ArgumentParser(description="Process input path and output directory.")

parser.add_argument("-i", "--input", required=True, 
                    help="Input path")
parser.add_argument("-o", "--output", required=True, 
                    help="Output directory")
parser.add_argument(
    "-c", "--config",
    default="config.txt",
    help="Configuration file path (default: config.txt)"
)
args = parser.parse_args()

rads = ['CDV', 'LMI', 'PDA', 'PBE']
for r in rads:
    if r in args.input:
        RADAR = r
        break

vols = ['VOLA', 'VOLB', 'VOLC']
for v in vols:
    if v in args.input:
        VOLUME = v
        break

# Create logs folder in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging to write to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, f'mod2_{RADAR}.log')),
        logging.StreamHandler()
    ]
)

# Set DEBUG level for application loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.info("")
logger.info("=" * 80)
logger.info("MODULE 2 started")
logger.info("=" * 80)

# Check if path exists and log accordingly
if os.path.exists(args.input): logger.debug(f"Input path: {args.input}")
else: 
    logger.error(f"Input path does not exist: {args.input}")
    raise

if os.path.exists(args.output): logger.debug(f"Output path: {args.output}")
else:
    logger.error(f"Output directory does not exist: {args.output}")
    raise

if os.path.exists(args.config): logger.debug(f"Configuration file path: {args.config}")
else:
    logger.error(f"Configuration file does not exist: {args.config}")
    raise

def load_config(config_file: str) -> dict:

    # Read the configuration file into a single string. Propagate a
    # clear FileNotFoundError if the file cannot be opened.
    try:
        with open(config_file, "r") as f:
            config_data = f.read()
    except Exception as e:
        logger.error(f"Configuration file '{config_file}' not found or cannot be read: {e}")
        raise FileNotFoundError(
            "Configuration file 'config.txt' not found. Please create it based on 'config_template.txt'."
        )

    # Parse and validate expected configuration values from specific lines.
    config_lines = config_data.split("\n")
    config = {}
    for l in range(1,len(config_lines)+1):
        line = config_lines[l-1] # Adjust for 0-based index
        
        # Parse expected configuration values based on line number
        if l == 5:
            config["PROD_types"] = line.strip().split(", ")
            if not all(comp in ["LUE", "CAPPI"] for comp in config["PROD_types"]):
                logger.error(f"Invalid PRODUCT TYPES in config: {config['PROD_types']}")
                raise ValueError("PRODUCT TYPES in config.txt must be a comma-separated list of 'LUE' and/or 'CAPPI'")
        
        elif l == 11: config["CAPPI_H"] = line.strip()
        elif l == 14: 
            config["dl"] = line.strip()
            try:
                config["CAPPI_H"] = int(config["CAPPI_H"])
                config["dl"] = int(config["dl"])
            except:
                logger.error(f"Invalid CAPPI HEIGHT or CARTESIAN RESOLUTION: {config['CAPPI_H']}, {config['dl']}")
                raise ValueError("CAPPI HEIGHT and CARTESIAN RESOLUTION in config.txt must be an integer value in meters.")

        if l == 17: config["SR_DEM_path"] = line.strip()
        elif l == 20: 
            config["LR_DEM_path"] = line.strip()
            try:
                with open(config["SR_DEM_path"], "r") as f:
                    pass
                with open(config["LR_DEM_path"], "r") as f:
                    pass
            except:
                logger.error(f"DEM file paths are incorrect: SR={config['SR_DEM_path']}, LR={config['LR_DEM_path']}")
                raise ValueError("DEM file path(s) in config.txt is/are incorrect.")

    return config

elevs = np.arange(0, 90, 0.01)
for e in elevs:
    if e in elevs:
        pass

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

def import_DEM(DEM_path):
    # Import DEM data
    try:
        with rasterio.open(DEM_path) as src:
            DEM_values = src.read(1)
            height, width = src.shape
            transform = src.transform
    except Exception as e:
        logger.error(f"Error reading DEM file: {e}")
        raise
    
    try:
        x = np.arange(width) * transform.a + transform.c
        y = np.arange(height) * transform.e + transform.f
        DEM_coords = np.array(np.meshgrid(x, y))
    except Exception as e:
        logger.error(f"Error computing DEM coordinates: {e}")
        raise
    
    return DEM_values, np.moveaxis(DEM_coords, 0, 2), transform

def get_DEM_resampled(DEM_values, DEM_transform, xgrid, ygrid, dl, save_path):
    # --- 1. Check if file exists ---
    if os.path.exists(save_path):
        try:
            data = np.load(save_path)
            DEM_saved = data["DEM"]
            x_saved = data["xgrid"]
            y_saved = data["ygrid"]

            # --- 2. Check if grids match ---
            if (np.array_equal(x_saved, xgrid) and 
                np.array_equal(y_saved, ygrid)):
                
                logger.info(f"Loaded existing resampled DEM from cache: {save_path}")
                return DEM_saved

        except Exception as e:
            logger.warning(f"Could not read saved DEM ({e}).")

    # --- 3. Compute resampled DEM ---
    logger.debug("Computing DEM resampling...")
    
    x_min, x_max = xgrid.min(), xgrid.max()
    y_min, y_max = ygrid.min(), ygrid.max()
    
    new_transform = from_origin(x_min, y_max, dl, dl)
    dst_shape = (len(ygrid), len(xgrid))
    
    DEM_resampled = np.empty(dst_shape, dtype=np.float32)

    try:
        reproject(
            source=DEM_values,
            destination=DEM_resampled,
            src_transform=DEM_transform,
            src_crs="EPSG:4326",
            dst_transform=new_transform,
            dst_crs="EPSG:25831",
            resampling=Resampling.nearest
        )
    except Exception as e:
        logger.error(f"Error during DEM resampling: {e}")
        raise

    # --- 4. Save result ---
    try:
        np.savez_compressed(
            save_path,
            DEM=DEM_resampled,
            xgrid=xgrid,
            ygrid=ygrid
        )
        logger.info(f"Resampled DEM saved to disk: {save_path}")
    except Exception as e:
        logger.error(f"Error saving resampled DEM: {e}")
        raise

    return DEM_resampled

t0 = time.time()

# Load configuration parameters from "config" file
config_file = args.config
try:
    config = load_config(config_file)
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise
PROD_types = config["PROD_types"]
CAPPI_H = config["CAPPI_H"]
dl = config["dl"]

# Open the input dataset (previously processed by MODULE 1)
try:
    ds_input = xr.open_dataset(args.input)
except Exception:
    logger.error("Input dataset can not be opened by xarray.")
    raise

# Change coordinate type from polar (range, azimuth) to cartesian (x, y)
logger.info("Computing coordinate transformation...")
cartesian_dic = {}
for elev in ds_input.elev.values:
    try:
        cartesian_dic[elev] = polar2cartesian(ds_input.sel(elev=elev), dl)
        xgrid, ygrid = cartesian_dic[elev]['xgrid'], cartesian_dic[elev]['ygrid']

    except Exception as e:
        logger.error(f"An error ocurred during coordinate transformation of elevation {elev}:")
        logger.error(e)
        raise

# Create products wanted (CAPPI and/or LUE)
if "CAPPI" in PROD_types:
    # Apply height-to-CAPPI quality index
    CAPPI_QI_elev = []
    for elev in cartesian_dic:
        elev_dic = cartesian_dic[elev]
        Z_e = elev_dic["Z"]
        QI_e = elev_dic["QI"]
        H_to_CAPPI = np.abs(elev_dic["H"] - CAPPI_H)
        QI_e[Z_e != -32] = QI_e[Z_e != -32] * distance_weighting(H_to_CAPPI)[Z_e != -32]
        CAPPI_QI_elev.append(QI_e)
    
    Z_arr = np.array([cartesian_dic[elev_dic]["Z"] for elev_dic in cartesian_dic])
    H_arr = np.array([cartesian_dic[elev_dic]["H"] for elev_dic in cartesian_dic])
    ds_CAPPI = xr.Dataset({"Z": (["elev", "y", "x"], Z_arr), 
                           "QI": (["elev", "y", "x"], CAPPI_QI_elev), 
                           "H": (["elev", "y", "x"], H_arr)}, 
                           coords={"elev": ds_input.elev.values, "x": xgrid, "y": ygrid})
    
    # Compute single-radar CAPPI products
    CAPPI_Z, CAPPI_QI, CAPPI_ELEV = make_CAPPI(ds_CAPPI, CAPPI_H)

if "LUE" in PROD_types:
    SR_LR = 'LR' if VOLUME == 'VOLA' else 'SR'
    DEM_key = f"{SR_LR}_DEM_path"
    DEM_values, DEM_coords, DEM_transform = import_DEM(config[DEM_key])
    DEM_resampled = get_DEM_resampled(DEM_values, DEM_transform, xgrid, ygrid, 
                                      dl, os.path.join(script_dir, f"{SR_LR}_DEM_resampled.npz"))

    # Apply height-to-ground quality index
    LUE_QI_elev = []
    for elev in cartesian_dic:
        elev_dic = cartesian_dic[elev]
        Z_e = elev_dic["Z"]
        QI_e = elev_dic["QI"]
        H_to_ground = elev_dic["H"] - DEM_resampled
        QI_e[Z_e != -32] = QI_e[Z_e != -32] * distance_weighting(H_to_ground)[Z_e != -32]
        LUE_QI_elev.append(QI_e)

    Z_arr = np.array([cartesian_dic[elev_dic]["Z"] for elev_dic in cartesian_dic])
    H_arr = np.array([cartesian_dic[elev_dic]["H"] for elev_dic in cartesian_dic])
    ds_LUE = xr.Dataset({"Z": (["elev", "y", "x"], Z_arr), 
                         "QI": (["elev", "y", "x"], LUE_QI_elev), 
                         "H": (["elev", "y", "x"], H_arr)}, 
                         coords={"elev": ds_input.elev.values, "x": xgrid, "y": ygrid})
    
    # Compute and store single-radar LUE products
    LUE_Z, LUE_QI, LUE_ELEV = make_LUE(ds_LUE, DEM_resampled)

# Create CAPPI dataset if requested
if "CAPPI" in PROD_types:
    cappi_ds = xr.Dataset(
        {
            "Z": (["elev", "y", "x"], Z_arr),
            "H": (["elev", "y", "x"], H_arr),
            "CAPPI_Z": (["y", "x"], CAPPI_Z),
            "CAPPI_Q": (["y", "x"], CAPPI_QI),
            "CAPPI_ELEV": (["y", "x"], CAPPI_ELEV),
        },
        coords={
            "elev": ds_input.elev.values,
            "x": xgrid,
            "y": ygrid
        },
    )
    cappi_attributes = {
        "Z": {"long_name": "Radar Reflectivity", "units": "dBZ"},
        "H": {"long_name": "Height of measurement", "units": "m"},
        "CAPPI_Z": {"long_name": "Constant Altitude PPI Reflectivity", "units": "dBZ"}, 
        "CAPPI_Q": {"long_name": "Constant Altitude PPI Quality Index", "units": "0-1"}, 
        "CAPPI_ELEV": {"long_name": "Constant Altitude PPI Elevation", "units": "deg"},
    }
    for var in cappi_attributes:
        if var in cappi_ds.data_vars:
            cappi_ds[var].attrs["long_name"] = cappi_attributes[var]["long_name"]
            cappi_ds[var].attrs["units"] = cappi_attributes[var]["units"]
            cappi_ds[var].attrs["altitude"] = f"{CAPPI_H} m"

# Create LUE dataset if requested
if "LUE" in PROD_types:
    lue_ds = xr.Dataset(
        {
            "Z": (["elev", "y", "x"], Z_arr),
            "H": (["elev", "y", "x"], H_arr),
            "LUE_Z": (["y", "x"], LUE_Z),
            "LUE_Q": (["y", "x"], LUE_QI),
            "LUE_ELEV": (["y", "x"], LUE_ELEV),
        },
        coords={
            "elev": ds_input.elev.values,
            "x": xgrid,
            "y": ygrid
        },
    )
    lue_attributes = {
        "Z": {"long_name": "Radar Reflectivity", "units": "dBZ"},
        "H": {"long_name": "Height of measurement", "units": "m"},
        "LUE_Z": {"long_name": "Lowest Usable Elevation Reflectivity", "units": "dBZ"}, 
        "LUE_Q": {"long_name": "Lowest Usable Elevation Quality Index", "units": "0-1"}, 
        "LUE_ELEV": {"long_name": "Lowest Usable Elevation", "units": "deg"},
    }
    for var in lue_attributes:
        if var in lue_ds.data_vars:
            lue_ds[var].attrs["long_name"] = lue_attributes[var]["long_name"]
            lue_ds[var].attrs["units"] = lue_attributes[var]["units"]

dt_time = dt.datetime.strptime(ds_input.attrs["observation_period"], "%Y-%m-%dT%H:%MZ")
timestamp = dt_time.strftime("%Y%m%d%H%M")

t1 = time.time()
processing_time = f"{np.round(t1 - t0, 2)} s"

# Save CAPPI dataset if it was created
if "CAPPI" in PROD_types:
    cappi_ds.attrs["observation_period"] = ds_input.attrs["observation_period"]
    cappi_ds.attrs["processing_time"] = processing_time
    cappi_file = f"{args.output}/CPI_{RADAR}_{VOLUME}_{timestamp}.nc"
    cappi_ds.to_netcdf(cappi_file, engine="scipy")
    logger.info(f"CAPPI file saved to: {cappi_file}")

# Save LUE dataset if it was created
if "LUE" in PROD_types:
    lue_ds.attrs["observation_period"] = ds_input.attrs["observation_period"]
    lue_ds.attrs["processing_time"] = processing_time
    lue_file = f"{args.output}/LUE_{RADAR}_{VOLUME}_{timestamp}.nc"
    lue_ds.to_netcdf(lue_file, engine="scipy")
    logger.info(f"LUE file saved to: {lue_file}")

logger.info(f"Processing completed in {processing_time}")