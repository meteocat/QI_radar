from AutoPlot_tools import plot_composite_file

import argparse
import os
import logging
import xarray as xr
import datetime as dt
import geopandas as gpd
import time
import numpy as np
import rasterio

def read_geotiff_composite(geotiff_path):
    """Read a GeoTIFF composite file and extract coordinates and bands.
    
    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTIFF file.
    
    Returns
    -------
    dict
        Dictionary with 'x', 'y', 'bands' (dict of band_name -> data),
        'data_vars' (list), and 'attributes' (dataset-level tags).
    """
    with rasterio.open(geotiff_path) as src:
        # Extract coordinates from affine transform
        transform = src.transform
        x_res = transform.a
        y_res = transform.e
        x_min = transform.c
        y_max = transform.f
        
        # Reconstruct coordinate arrays
        x_vals = np.arange(x_min + x_res / 2, x_min + x_res / 2 + x_res * src.width, x_res)
        y_vals = np.arange(y_max + y_res / 2, y_max + y_res / 2 + y_res * src.height, y_res)
        
        # Read all bands and their descriptions
        bands_dict = {}
        data_vars = []
        for i in range(1, src.count + 1):
            var_name = src.descriptions[i - 1]
            if var_name:
                band_data = src.read(i).astype(np.float32)
                bands_dict[var_name] = band_data
                data_vars.append(var_name)
        
        # Read dataset-level attributes
        attributes = src.tags()
        
        return {
            'x': x_vals,
            'y': y_vals,
            'bands': bands_dict,
            'data_vars': data_vars,
            'attributes': attributes
        }

def create_save_path(out_dir, comp, prod, volume, timestamp):
    os.makedirs(f"{out_dir}/{volume}", exist_ok=True)
    os.makedirs(f"{out_dir}/{volume}/{prod}", exist_ok=True)
    todate_dir = f"{out_dir}/{volume}/{prod}/{comp}"
    os.makedirs(todate_dir, exist_ok=True)
    time_dt = dt.datetime.strptime(timestamp, '%Y%m%d%H%M')
    yy, mm, dd = time_dt.strftime("%Y"), time_dt.strftime("%m"), time_dt.strftime("%d")
    os.makedirs(f"{todate_dir}/{yy}", exist_ok=True)
    os.makedirs(f"{todate_dir}/{yy}/{mm}", exist_ok=True)
    save_dir = f"{todate_dir}/{yy}/{mm}/{dd}"
    os.makedirs(save_dir, exist_ok=True)

    return os.path.join(save_dir, f"{volume}_{prod}_{comp}_{timestamp}.png")

# Read command-line arguments
parser = argparse.ArgumentParser(description="Process input path and output directory.")

parser.add_argument("-i", "--input", required=True, 
                    help="Input path")
parser.add_argument("-o", "--output", required=True, 
                    help="Output directory")
parser.add_argument(
    "-s", "--shapefile", required=True,
    help="Shapefile path"
)
args = parser.parse_args()

vols = ['VOLA', 'VOLB', 'VOLC']
for v in vols:
    if v in args.input:
        VOLUME = v
        break

# Create logs folder in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure logging to write to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(script_dir, f'mod4.log')),
        logging.StreamHandler()
    ]
)

# Set DEBUG level for application loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.info("")
logger.info("=" * 80)
logger.info("MODULE 4 started")
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

if os.path.exists(args.shapefile): logger.debug(f"Shapefile path: {args.shapefile}")
else:
    logger.error(f"Shapefile does not exist: {args.shapefile}")
    raise

t0 = time.time()

try:
    comarques = gpd.read_file(args.shapefile).to_crs(epsg=25831)
    
    # Determine if input is GeoTIFF or NetCDF and read accordingly
    if args.input.endswith('.tif'):
        logger.info("Reading GeoTIFF file")
        geotiff_data = read_geotiff_composite(args.input)
        x_vals = geotiff_data['x']
        y_vals = geotiff_data['y']
        bands_dict = geotiff_data['bands']
        data_vars = geotiff_data['data_vars']
        attributes = geotiff_data['attributes']
        dt_time = dt.datetime.strptime(attributes["observation_period"], "%Y-%m-%dT%H:%MZ")
    else:
        logger.info("Reading NetCDF file")
        with xr.open_dataset(args.input, engine="scipy") as ds:
            x_vals = ds.x.values
            y_vals = ds.y.values
            data_vars = list(ds.data_vars)
            bands_dict = {var: ds[var].values for var in data_vars}
            attributes = dict(ds.attrs)
            dt_time = dt.datetime.strptime(attributes["observation_period"], "%Y-%m-%dT%H:%MZ")
    
    timestamp = dt_time.strftime("%Y%m%d%H%M")
    
    # Extract unique composite-product combinations
    for el in (list(set(['_'.join(var.split('_')[:-1]) for var in data_vars]))):
        comp, prod = el.split('_')
                    
        Z = bands_dict[f'{comp}_{prod}_Z']
        Q = bands_dict[f'{comp}_{prod}_QI']
        RAD = bands_dict[f'{comp}_{prod}_RAD']
        ELEV = bands_dict[f'{comp}_{prod}_ELEV']
        
        if prod == 'CAPPI':
            prod = 'CAPPI1.0km'
        save_path = create_save_path(args.output, comp, prod, VOLUME, timestamp)
        logger.info(f"Creating file image: {VOLUME}_{prod}_{comp}_{timestamp}.png")

        plot_composite_file(Z, Q, RAD, ELEV, x_vals, y_vals, save_path, comarques)

    t1 = time.time()
    logger.info(f"Finished creating file image in {t1-t0:.2f} seconds")
    logger.info(f"Output saved to: {args.output}")
except Exception as e:
    logger.error(f"An error occurred during processing: {e}")
    raise