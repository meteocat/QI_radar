from quality_tools import single_PPI, get_processor_data
import argparse
import datetime as dt
import xradar as xd
import rasterio
import numpy as np
import os
import xarray as xr
import time
import logging

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

# Create logs folder in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging to write to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, f'mod1_{RADAR}.log')),
        logging.StreamHandler()
    ]
)

# Set DEBUG level for application loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.info("")
logger.info("=" * 80)
logger.info("MODULE 1 started")
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
        
        elif l == 23: 
            config["TOP12_clim_path"] = line.strip()
            try:
                with open(config["TOP12_clim_path"], "r") as f:
                    pass
                break
            except:
                logger.error(f"TOP12 climatology file path is incorrect: {config['TOP12_clim_path']}")
                raise ValueError("TOP12 climatology file path in config.txt is incorrect.")
    
    return config

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

t0 = time.time()

# Load configuration parameters from "config" file
config_file = args.config
try:
    config = load_config(config_file)
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise
TOP12_clim_path = config["TOP12_clim_path"]

# Load IRIS
IRIS_path = args.input
try:
    vol = xd.io.open_iris_datatree(IRIS_path, reindex_angle=False)
    scan_name = vol.attrs['scan_name']

    for v in ['A', 'B', 'C']:
        if v in scan_name:
            VOLUME = f"VOL{v}"
            break

    logger.debug(f"Scan name: {scan_name} ({VOLUME})")
except Exception as e:
    logger.error(f"Error opening IRIS file: {e}")
    raise

try:
    ts = vol.time_coverage_start.values
    dt_time = dt.datetime.fromisoformat(ts.item().replace('Z', '+00:00'))
    yy, mm, dd, hh, MM = dt_time.year, dt_time.month, dt_time.day, dt_time.hour, dt_time.minute
    IRIS_time = (yy, mm, dd, hh, MM)
    logger.info(f"Radar data timestamp: {yy}-{mm:02d}-{dd:02d} {hh:02d}:{MM:02d}")
except Exception as e:
    logger.error(f"Error extracting timestamp: {e}")
    raise

DEM_key = f"{'L' if VOLUME == 'VOLA' else 'S'}R_DEM_path"
DEM_values, DEM_coords, DEM_transform = import_DEM(config[DEM_key])

# Create output name
dt_time = dt_time.astimezone(dt.timezone.utc).replace(tzinfo=None)
possible_timestamps = np.arange(dt.datetime(dt_time.year, dt_time.month, dt_time.day, 0, 0), dt.datetime(dt_time.year, dt_time.month, dt_time.day, 23, 59), dt.timedelta(minutes=6))
closest_timestamp = possible_timestamps[np.argmin(np.abs(possible_timestamps - np.array([dt_time]).astype('datetime64[ns]')))].astype(dt.datetime)
timestamp = closest_timestamp.strftime("%Y%m%d%H%M")
output_file = f"{args.output}/PPI_{RADAR}_{VOLUME}_{timestamp}.nc"

try:
    logger.info("Starting Quality Index computation.")

    # Extract radar calibration and processing constants
    instr_var = get_processor_data(IRIS_path)
    vol = xd.io.open_iris_datatree(IRIS_path, reindex_angle=False)

    # Initialize lists to store results and set sweep list
    N_sweeps = len(vol["sweep_fixed_angle"].values)
    sweep_list = np.arange(N_sweeps)

    logger.info(f"Elevations to process: {' - '.join([str(a) for a in vol['sweep_fixed_angle'].values])}")

    # Compute PPIs for each sweep
    for sweep in sweep_list:
        # Extract sweep dataset
        ds = vol[f"sweep_{sweep}"]

        # Process single PPI and append results to lists
        qual_out = single_PPI(ds, TOP12_clim_path, DEM_values, DEM_coords, instr_var)

        if sweep == sweep_list[0]:
            arr_shape = (N_sweeps, len(qual_out["QDET"][:,0]), len(qual_out["QDET"][0,:]))
            output_ds = xr.Dataset(
                {
                    "Z": (["elev", "azimuth", "range"], np.ones(arr_shape) * np.nan),
                    "Q": (["elev", "azimuth", "range"], np.ones(arr_shape) * np.nan),
                    "QDET": (["elev", "azimuth", "range"], np.ones(arr_shape) * np.nan),
                    "QNDET": (["elev", "azimuth", "range"], np.ones(arr_shape) * np.nan),
                    "QCL": (["elev", "azimuth", "range"], np.ones(arr_shape) * np.nan),
                    "QDeltaZ": (["elev", "azimuth", "range"], np.ones(arr_shape) * np.nan),
                    "QOS": (["elev", "azimuth", "range"], np.ones(arr_shape) * np.nan),
                    "QFI": (["elev", "azimuth", "range"], np.ones(arr_shape) * np.nan),
                    "QMDR": (["elev", "azimuth", "range"], np.ones(arr_shape) * np.nan),
                    "sweep_mode": ds.sweep_mode.values,
                },
                coords={
                    "elev": vol["sweep_fixed_angle"].values,
                    "range": ds.range.values,
                    "azimuth": ds.azimuth.values,
                    "elevation": (["elev", "azimuth"], 
                                  np.ones((
                                      len(vol["sweep_fixed_angle"].values), 
                                      len(ds.elevation.values)
                                  ))* np.nan),
                    "longitude": ds.longitude.values,
                    "latitude": ds.latitude.values,
                    "altitude": ds.altitude.values,
                },
            )
        
        for ind in qual_out:
            output_ds[ind][sweep, ...] = qual_out[ind]
        output_ds["Z"][sweep, ...] = ds.DBZH.values
        output_ds["elevation"][sweep, :] = ds.elevation.values

    attributes = {
            "Z": {"long_name": "Corrected Reflectivity", "units": "dBZ"},
            "Q": {"long_name": "Total Quality Index", "units": "0-1"},
            "QDET": {"long_name": "Quality Index of Detected Precipitation Region", "units": "0-1"},
            "QNDET": {"long_name": "Quality Index of Undetected Precipitation Region", "units": "0-1"},
            "QCL": {"long_name": "Ground Clutter Quality Index", "units": "0-1"},
            "QDeltaZ": {"long_name": "Reflectivity Accuracy Quality Index", "units": "0-1"},
            "QOS": {"long_name": "Precipitation Overshooting Quality Index", "units": "0-1"},
            "QFI": {"long_name": "Processor Filtering Quality Index", "units": "0-1"},
            "QMDR": {"long_name": "Minimum Detectable Reflectivity Quality Index", "units": "0-1"},
        }
    for var in attributes:
        if var in output_ds.data_vars:
            output_ds[var].attrs["long_name"] = attributes[var]["long_name"]
            output_ds[var].attrs["units"] = attributes[var]["units"]

    # Add processing time and status
    t1 = time.time()
    output_ds.attrs["processing_time"] = f"{np.round(t1 - t0, 2)} s"
    output_ds.attrs["observation_period"] = closest_timestamp.strftime("%Y-%m-%dT%H:%MZ")
    output_ds.to_netcdf(output_file, engine="scipy")

    logger.info(f"Processing completed in {output_ds.attrs['processing_time']}")
    logger.info(f"File saved to: {output_file}")

except Exception as e:
    error_message = str(e)
    logger.error(f"Error during processing: {error_message}")