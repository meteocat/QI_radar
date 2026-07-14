
# Radar Quality Index Processing

This repository implements a radar processing workflow for generating quality-aware Cartesian products from polar radar volumes. The pipeline is designed for IRIS-format radar data and produces single-radar products as well as multi-radar composites for operational meteorology.

## Overview

The project processes raw radar reflectivity data in four stages:

1. Estimate per-gate quality indices from radar processor information, beam geometry, clutter effects, overshooting, and attenuation-related uncertainty.
2. Transform polar radar data into Cartesian grids and build single-radar CAPPI or LUE products.
3. Combine results from multiple radars into composite products.
4. Render the resulting composites as map-based PNG images.

The main purpose is to create reflectivity fields that are not only geographically transformed, but also weighted by a quality index that reflects how reliable each pixel is.

## What the project produces

The workflow generates:

- Per-radar polar quality products in NetCDF format.
- Single-radar CAPPI or LUE products in NetCDF format.
- Multi-radar composite GeoTIFF products.
- Plot images in PNG format for visualization.

Typical output variables include:

- `Z`: reflectivity in dBZ
- `QI`: quality index in the range $[0, 1]$
- `RAD`: radar identifier selected by the composite logic
- `ELEV`: elevation angle used for the selected product

## How the pipeline works

The workflow is organized into separate modules that should be executed in order.

### 1. Quality generation

The first stage reads an IRIS radar file and computes several quality-related fields for each sweep:

- `QDET`: quality for detected precipitation regions
- `QNDET`: quality for undetected regions
- `QCL`: clutter-related quality
- `QDeltaZ`: reflectivity accuracy quality
- `QOS`: overshooting-related quality
- `QFI`: processor filtering quality
- `QMDR`: minimum detectable reflectivity quality

These values are stored in a NetCDF file together with the original reflectivity and sweep metadata.

### 2. Product generation

The second stage converts the polar PPI data to Cartesian coordinates and builds either:

- a CAPPI product, which selects the best reflectivity and quality values at a fixed altitude, or
- an LUE product, which selects the lowest usable elevation that satisfies a quality threshold after accounting for terrain height.

### 3. Composite generation

The third stage combines single-radar CAPPI or LUE products from several radars into a network-wide composite. The composite methods supported by the code are:

- `MAXZ`: prefer the pixel with the highest reflectivity
- `MAXQI`: prefer the pixel with the highest quality index
- `MAXQCOND`: apply a more specialized rule that accounts for detection/undetection regimes and quality thresholds

### 4. Visualization

The final stage reads the composite GeoTIFF or NetCDF product and plots it over a shapefile-defined region, producing a PNG image with reflectivity, quality, radar source, and elevation panels.

## Repository structure

The current repository layout is:

```text
QI_radar/
├── 1_quality_generation/
│   ├── generate_quality_PPI.py
│   ├── quality_tools.py
│   └── logs/
├── 2_product_generation/
│   ├── generate_products.py
│   ├── polar2cartesian_tools.py
│   ├── CAPPI_LUE_tools.py
│   └── logs/
├── 3_composite_generation/
│   ├── generate_composite.py
│   ├── Composite_tools.py
│   └── logs/
├── 4_visualization/
│   ├── plot_products.py
│   └── AutoPlot_tools.py
├── config.txt
├── config_template.txt
├── requirements.txt
└── README.md
```

### Important files

- `1_quality_generation/generate_quality_PPI.py`: reads raw IRIS radar data and writes the polar-quality NetCDF files.
- `1_quality_generation/quality_tools.py`: contains the core quality-index computation logic.
- `2_product_generation/generate_products.py`: creates CAPPI or LUE products from the quality-processed PPI data.
- `2_product_generation/polar2cartesian_tools.py`: performs the polar-to-Cartesian transformation.
- `2_product_generation/CAPPI_LUE_tools.py`: builds CAPPI and LUE fields from the transformed elevation stacks.
- `3_composite_generation/generate_composite.py`: combines single-radar products into composites and writes GeoTIFF output.
- `3_composite_generation/Composite_tools.py`: implements the composite selection strategies.
- `4_visualization/plot_products.py`: produces PNG plots from the composite outputs.

## Installation

The project depends on several scientific Python packages.

### Requirements

This repository expects Python 3.10 or newer. The current dependency set is defined in `requirements.txt` and includes:

- `numpy`
- `xarray`
- `netCDF4`
- `rasterio`
- `wradlib`
- `xradar`
- `cartopy`
- `geopandas`
- `matplotlib`
- `pyproj`
- `shapely`

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

If you are using a virtual environment, create and activate it before installing the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configuration

The main configuration file is `config.txt`. A template is available in `config_template.txt`.

You should update the following values before running the workflow:

- `PRODUCT TYPES`: choose `LUE`, `CAPPI`, or both
- `COMPOSITE TYPES`: choose `MAXZ`, `MAXQI`, `MAXQCOND`, or any combination
- `CAPPI HEIGHT (meters)`: altitude used for the CAPPI product
- `CARTESIAN RESOLUTION (meters)`: grid spacing for the Cartesian products
- `REGION SHAPEFILE PATH`: path to the shapefile used by the plotting step
- `SHORT-RANGE DEM DIRECTORY PATH`: DEM used for short-range processing
- `LONG-RANGE DEM DIRECTORY PATH`: DEM used for long-range processing
- `ECHO TOPS 12dBZ CLIMATOLOGY PATH`: climatology file used for overshooting quality estimation

The file format is strict and should be preserved. The current repository already contains a working example configuration in `config.txt`.

## Running the project

The scripts are intentionally split by processing stage. Run them in order.

### 1. Generate quality products

From the repository root, run:

```bash
python 1_quality_generation/generate_quality_PPI.py \
  -i /path/to/radar_file.raw \
  -o /path/to/output_dir \
  -c config.txt
```

This produces a NetCDF file named like:

```text
PPI_<RADAR>_<VOLUME>_<YYYYMMDDHHMM>.nc
```

### 2. Generate CAPPI or LUE products

Use the NetCDF file from step 1 as input:

```bash
python 2_product_generation/generate_products.py \
  -i /path/to/PPI_<RADAR>_<VOLUME>_<YYYYMMDDHHMM>.nc \
  -o /path/to/output_dir \
  -c config.txt
```

This writes one or more NetCDF files named like:

```text
CPI_<RADAR>_<VOLUME>_<YYYYMMDDHHMM>.nc
LUE_<RADAR>_<VOLUME>_<YYYYMMDDHHMM>.nc
```

### 3. Create radar composites

Provide one of the single-radar product files from step 2 as input. The script will look for matching files from the other radars and build a composite:

```bash
python 3_composite_generation/generate_composite.py \
  -i /path/to/LUE_<RADAR>_<VOLUME>_<YYYYMMDDHHMM>.nc \
  -o /path/to/output_dir \
  -c config.txt
```

The output is a GeoTIFF named like:

```text
CMP_LUE_<VOLUME>_<YYYYMMDDHHMM>.tif
```

or:

```text
CMP_CPI_<VOLUME>_<YYYYMMDDHHMM>.tif
```

### 4. Create plots

To generate plot images for a composite product:

```bash
python 4_visualization/plot_products.py \
  -i /path/to/CMP_LUE_<VOLUME>_<YYYYMMDDHHMM>.tif \
  -o /path/to/plot_dir \
  -s /path/to/region.shp
```

The plotting step writes PNG files into a nested directory structure under the chosen output directory. The script creates one folder for the volume, then one for the product type, then one for the composite type, and finally year/month/day folders before writing the PNG file.

```text
<output_dir>/
└── <VOLUME>/
    └── <PRODUCT>/
        └── <COMPOSITE>/
            └── <YYYY>/
                └── <MM>/
                    └── <DD>/
                        └── <VOLUME>_<PRODUCT>_<COMPOSITE>_<YYYYMMDDHHMM>.png
```

For example, a CAPPI composite for volume VOLA could be written to a path like:

```text
<output_dir>/VOLA/CAPPI/MAXZ/2026/07/14/VOLA_CAPPI_MAXZ_202607141200.png
```

## Outputs

The repository writes outputs in three formats:

### NetCDF files

Produced by stages 1 and 2. These are used as intermediate data products and contain radar-native metadata, reflectivity, and quality information.

### GeoTIFF files

Produced by stage 3. Composite products are written as multiband GeoTIFF files where each band corresponds to a variable such as reflectivity, quality index, radar identifier, or elevation.

### PNG files

Produced by stage 4. These are visualization products that include reflectivity, quality, radar source, and elevation panels.

## Notes and limitations

- The code expects radar input in the IRIS format and uses the radar metadata embedded in the file header.
- The workflow is currently organized as a set of stage-specific scripts rather than a single entry-point program.
- The processing time can be significant, especially for large Cartesian grids and for multi-radar composites.
- The output filenames follow the radar and volume naming conventions detected from the input path.
- The visualization step requires a valid shapefile in EPSG:25831-compatible coordinates or a compatible CRS conversion path.

## Performance considerations

The most expensive operations are:

- polar-to-Cartesian remapping,
- DEM resampling,
- quality computation over many range bins and rays, and
- composite generation across multiple radar files.

Using larger Cartesian grid spacings and limiting the number of requested products can reduce runtime substantially.
