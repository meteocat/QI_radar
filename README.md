
# **Radar Quality Index Processing**

This repository contains the full processing scripts developed for the **Quality Index (Q)** applied to weather radar reflectivity data from the XRAD C‑band radar network of the *Servei Meteorològic de Catalunya (SMC)*. The software computes gate‑level quality indicators, transforms them into Cartesian radar products, and generates network‑wide composites suitable for operational meteorology.

---

## **Repository Structure**

The repository structure is as follows:

```
QI_radar/
├── MAIN.py                         # Main script orchestrating the pipeline
├── Import_config.py                # Loads and parses the configuration file
├── FindIRISFiles.py                # Search and locate IRIS radar data files
├── Polar2Cartesian_PPI.py          # Polar to Cartesian conversion of PPI radar data
├── Composite_tools.py              # Compositing radar data from multiple radars
├── CAPPI_LUE_tools.py              # Tools for generating CAPPI and LUE products
|
├── config_template.txt             # Configuration file with processing parameters
├── HIST_TOP12.nc                   # Climatological echo tops data file
├── README.md                       # This documentation file
|
└── visualization/                  # Directory for visualization outputs
```

---

## **Data Format**

### ▶ **Input Files**

The pipeline requires the following input data to be configured in a new configuration file named `config.txt` (take example from the `config_template.txt`):

- **Raw Radar Data Files**: IRIS format (.RAW) files containing polar radar reflectivity data from the XRAD C-band radar network. These files are located in the directory specified in `config.txt` (RAW DATA DIRECTORY).

- **Auxiliary Data**:
  - DEM files (GeoTIFF format) for terrain correction, specified in `config.txt`.
  - Climatological echo tops data (NetCDF format) for quality index computation (`HIST_TOP12.nc`).

- **Configuration File**: `config.txt` - A text file specifying processing parameters, including:
  - Initial and final UTC times for processing (note that the final time is not processed).
  - Volume scan type (VOLA, VOLB, or VOLBC). Choose according to the desired products.
  - CAPPI height in meters.
  - Cartesian grid resolution in meters. Note that modifying this parameter will significantly affect processing time.
  - Raw data directory path.
  - Processed data directory path (where output products are saved).
  - Paths to Digital Elevation Model (DEM) files for short-range and long-range processing.
  - Temporal storage directory name for individual radar PPI fields.
  - Path to echo tops 12dBZ climatology file.

### ▶ **Output Files**

The pipeline generates the following composite products as NetCDF (.nc) files organized by volume type, product type, composite type, and date:

- **CAPPI (Constant Altitude Plan Position Indicator)**: Cartesian reflectivity fields at a specified height, with MAX-Z (maximum reflectivity) and MAX-QI (maximum quality index) composites.
- **LUE (Lowest Usable Elevation)**: Products from the lowest usable elevation angles, with MAX-Z and MAX-QI composites.

Each output file contains variables:
- `Z`: Reflectivity (dBZ)
- `QI`: Quality Index (dimensionless, 0-1)
- `RAD`: Radar identifier chosen by the composite criteria (integer)
- `ELEV`: Elevation angle used (degrees)

---

## **Visualization**

Visualization output may be generated using the ```plotComposite.py``` script following this command:

- If only wanting to visualise: ```python plotComposite.py [product_file_path.nc]```
- If wanting to save png: ```python plotComposite.py s [product_file_path.nc] [saving_directory]```