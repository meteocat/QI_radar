from Composite_tools import composite

import datetime as dt
import argparse
import os
import logging
import time
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import Affine

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
other_rads = [r for r in rads if r != RADAR]

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
        logging.FileHandler(os.path.join(logs_dir, f'mod3_{RADAR}.log')),
        logging.StreamHandler()
    ]
)

# Set DEBUG level for application loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def write_geotiff(ds: xr.Dataset, output_file: str, crs: str = "EPSG:25831"):
    """Write xarray Dataset as multiband GeoTIFF with metadata preservation.
    
    Each data variable is stored as a separate band. Dataset-level and 
    variable-level attributes are preserved as GeoTIFF metadata.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to write (must have x and y coordinates).
    output_file : str
        Output file path (.tif).
    crs : str
        Coordinate Reference System as EPSG code (default: EPSG:25831 for UTM).
    """
    # Extract coordinates
    x_coords = ds.x.values
    y_coords = ds.y.values
    
    # Compute pixel resolution
    x_res = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
    y_res = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0
    
    # Compute top-left corner (pixel center to corner transformation)
    x_min = float(x_coords[0] - x_res / 2)
    y_max = float(y_coords[0] - y_res / 2)
    
    # Create affine transform
    transform = Affine.translation(x_min, y_max) * Affine.scale(x_res, y_res)
    
    # Get data variables and dimensions
    data_vars = list(ds.data_vars)
    n_bands = len(data_vars)
    height, width = len(y_coords), len(x_coords)
    
    logger.debug(f"Writing GeoTIFF with {n_bands} bands, dimensions {height}x{width}")
    
    # Create and write GeoTIFF
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=n_bands,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        # Write each variable as a band
        for band_idx, var_name in enumerate(data_vars, start=1):
            # Extract and convert data to float32
            data = ds[var_name].values.astype(np.float32)
            
            # Write band data
            dst.write(data, band_idx)
            
            # Set band description (variable name)
            dst.set_band_description(band_idx, var_name)
            
            # Write band-level attributes/metadata
            var_attrs = ds[var_name].attrs
            if var_attrs:
                dst.update_tags(band_idx, **var_attrs)
            
            logger.debug(f"Band {band_idx}: {var_name}")
        
        # Write dataset-level attributes as GeoTIFF metadata
        if ds.attrs:
            dst.update_tags(**ds.attrs)
    
    logger.info(f"GeoTIFF written with {n_bands} bands to {output_file}")


logger.info("=" * 80)
logger.info("MODULE 3 started")
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
        if l == 8: 
            config["COMP_types"] = line.strip().split(", ")
            if not all(comp in ["MAXZ", "MAXQI", "MAXQCOND"] for comp in config["COMP_types"]):
                logger.error(f"Invalid COMPOSITE TYPES in config: {config['COMP_types']}")
                raise ValueError("COMPOSITE TYPES in config.txt must be a comma-separated list of 'MAXZ', 'MAXQI' and/or 'MAXQCOND'")
            
    return config

t0 = time.time()

# Detect product type from input filename
if "LUE_" in args.input:
    PROD_TYPE = "LUE"
    file_prefix = "LUE"
elif "CPI_" in args.input:
    PROD_TYPE = "CAPPI"
    file_prefix = "CPI"
else:
    logger.error("Input filename must contain 'LUE_' or 'CPI_' to identify product type")
    raise ValueError("Input filename must be formatted as 'LUE_RAD_VOLX_YYMMDDHHMM.nc' or 'CPI_RAD_VOLX_YYMMDDHHMM.nc'")

# Load configuration parameters from "config" file
config_file = args.config
try:
    config = load_config(config_file)
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise
COMP_types = config["COMP_types"]

input_dir, filename = os.path.split(args.input)

try:
    ds_input = xr.open_dataset(args.input)
except Exception:
    logger.error("Input dataset can not be opened by xarray.")
    raise

dt_time = dt.datetime.strptime(ds_input.attrs["observation_period"], "%Y-%m-%dT%H:%MZ")

# Search for other radar files with matching product type
other_rads_paths = [dt_time.strftime(f"{input_dir}/{file_prefix}_{o_r}_{VOLUME}_%Y%m%d%H%M.nc") for o_r in other_rads]

logger.info(f'Files used for the {PROD_TYPE} composite:')
logger.info(os.path.split(args.input)[1])
confirmed_paths = [args.input]
confirmed_rads = [RADAR]
for p in other_rads_paths:
    try:
        xr.open_dataset(p)
        confirmed_paths.append(p)
        logger.info(os.path.split(p)[1])

        for r in rads:
            if r in p:
                RAD = r
                break
        confirmed_rads.append(RAD)
    except:
        None

N_rad = len(confirmed_paths)
ds_x, ds_y = ds_input.x.values, ds_input.y.values
Z_ind_rad = np.ones((N_rad, len(ds_y), len(ds_x))) * np.nan       # Single-radar reflectivity
QI_ind_rad = np.ones((N_rad, len(ds_y), len(ds_x))) * np.nan      # Single-radar QI
ELEV_ind_rad = np.ones((N_rad, len(ds_y), len(ds_x))) * np.nan    # Single-radar ELEV

output_ds = xr.Dataset({}, coords={"x": ds_x, "y": ds_y})

i = 0
for p in confirmed_paths:
    try:
        ds_rad = xr.open_dataset(p)
        if PROD_TYPE == "CAPPI":
            Z_ind_rad[i, ...] = ds_rad.CAPPI_Z.values
            QI_ind_rad[i, ...] = ds_rad.CAPPI_Q.values
            ELEV_ind_rad[i, ...] = ds_rad.CAPPI_ELEV.values
        elif PROD_TYPE == "LUE":
            Z_ind_rad[i, ...] = ds_rad.LUE_Z.values
            QI_ind_rad[i, ...] = ds_rad.LUE_Q.values
            ELEV_ind_rad[i, ...] = ds_rad.LUE_ELEV.values
    except Exception as e:
        logger.error(f"Error loading data from {p}")

    i += 1

logger.info("Creating composites...")
for comp_type in COMP_types:
    Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP = composite(
        Z_ind_rad, QI_ind_rad, ELEV_ind_rad, comp_type
    )

    output_ds[f"{comp_type}_{PROD_TYPE}_Z"] = (("y", "x"), Z_COMP)
    output_ds[f"{comp_type}_{PROD_TYPE}_QI"] = (("y", "x"), QI_COMP)
    output_ds[f"{comp_type}_{PROD_TYPE}_RAD"] = (("y", "x"), RAD_COMP)
    output_ds[f"{comp_type}_{PROD_TYPE}_ELEV"] = (("y", "x"), ELEV_COMP)

if PROD_TYPE == "CAPPI": CAPPI_H = ds_input.CAPPI_Z.attrs['altitude']
else: CAPPI_H = None

dic =  {
    "composite_type": {
        "MAXZ": "Maximum Reflectivity",
        "MAXQI": "Maximum Quality Index",
        "MAXQCOND": "Maximum Quality Index under prescribed conditions",
    },
    "product_type": {
        "CAPPI": f"Constant Altitude Plan Position Indicator at {CAPPI_H}",
        "LUE": "Lowest Usable Elevation",
    },
    "variable": {
        "Z": "Reflectivity",
        "QI": "Quality Index",
        "RAD": "Radar used",
        "ELEV": "Elevation angle used"
    },
    "units": {
        "Z": "dBZ",
        "QI": "0-1",
        "RAD": "unitless",
        "ELEV": "degrees"
    }
}
for var in output_ds.data_vars:
    for c in dic["composite_type"]:
        if c in var:
            output_ds[var].attrs["composite_type"] = dic["composite_type"][c]
    for v in dic["variable"]:
        if v in var:
            output_ds[var].attrs["variable"] = dic["variable"][v]
            output_ds[var].attrs["units"] = dic["units"][v]
    # Add product type attribute
    output_ds[var].attrs["product_type"] = dic["product_type"][PROD_TYPE]

output_ds.attrs["radars"] = ", ".join(confirmed_rads)
dt_time = dt.datetime.strptime(ds_input.attrs["observation_period"], "%Y-%m-%dT%H:%MZ")
output_ds.attrs["observation_period"] = ds_input.attrs["observation_period"]
timestamp = dt_time.strftime("%Y%m%d%H%M")

# Output filename with product type prefix (CMP_CPI or CMP_LUE)
product_prefix = "CPI" if PROD_TYPE == "CAPPI" else "LUE"
output_file = f"{args.output}/CMP_{product_prefix}_{VOLUME}_{timestamp}.tif"

t1 = time.time()
output_ds.attrs["processing_time"] = f"{np.round(t1 - t0, 2)} s"
write_geotiff(output_ds, output_file)

logger.info(f"Processing completed in {output_ds.attrs['processing_time']}")
logger.info(f"File saved to: {output_file}")