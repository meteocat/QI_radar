
# **Radar Quality Index Processing**

This repository contains the full processing scripts developed for the **Quality Index (Q)** applied to weather radar reflectivity data from the XRAD C‑band radar network of the *Servei Meteorològic de Catalunya (SMC)*. The software computes gate‑level quality indicators, transforms them into Cartesian radar products, and generates network‑wide composites suitable for operational meteorology.

---

## **Repository Structure**

The repository structure is as follows:

```
qi_radar_products/
├── MAIN.py                         # Main script orchestrating the pipeline
├── Import_config.py                # Loads and parses the configuration file
├── FindIRISFiles.py                # Search and locate IRIS radar data files
├── Polar2Cartesian_PPI.py          # Polar to Cartesian conversion of PPI radar data
├── Composite_tools.py              # Compositing radar data from multiple radars
├── CAPPI_LUE_tools.py              # Tools for generating CAPPI and LUE products
|
├── config.txt                      # Configuration file with processing parameters
├── README.md                       # This documentation file
|
├── data/                           # Directory containing data files
│   └── raw/                        # Raw radar data files
│       └── ... (additional files)
|
└── visualization/                  # Directory for visualization outputs
```

---

## **Data Format**

### ▶ **Input Files**

The pipeline requires the following input data:

- **Raw Radar Data Files**: IRIS format (.RAW) files containing polar radar reflectivity data from the XRAD C-band radar network. These files are organized in subdirectories under `data/raw/`, named by radar station and date (e.g., `CDVRAW20250921/` for CDV radar on September 21, 2025). Each file corresponds to a volume scan at specific times (VOL-A, VOL-B, and VOL-C).

- **Auxiliary Data**:
  - DEM files (GeoTIFF format) for terrain correction.
  - Climatological echo tops data (NetCDF format) for quality index computation.

- **Configuration File**: `config.txt` - A text file specifying processing parameters, including:
  - Initial and final UTC times for processing (note that the final time is not processed).
  - Volume scan type (VOLA, VOLB, or VOLBC). Choose according to the desired products.
  - CAPPI height in meters.
  - Cartesian grid resolution in meters. Note that modifying this parameter will significantly affect processing time.
  - Paths to Digital Elevation Model (DEM) files for short-range and long-range processing.
  - Path to echo tops climatology data (NetCDF file).

### ▶ **Output Files**

The pipeline generates the following composite products as NetCDF (.nc) files:

- **CAPPI (Constant Altitude Plan Position Indicator)**: Cartesian reflectivity fields at a specified height, with MAX-Z (maximum reflectivity) and MAX-QI (maximum quality index) composites.
- **LUE (Lowest Usable Elevation)**: Products from the lowest usable elevation angles, with MAX-Z and MAX-QI composites.

Files are organized by volume type, product type, composite type, and date (e.g., `VOLB/CAPPI/MAXQI/2025/09/21/VOLB_CAPPI_MAXQI_2509211606.nc`).

Each output file contains variables:
- `Z`: Reflectivity (dBZ)
- `QI`: Quality Index (dimensionless, 0-1)
- `RAD`: Radar identifier chosen by the composite criteria (integer)
- `ELEV`: Elevation angle used (degrees)

Visualization outputs (plots, maps) may be generated in the `visualization/` directory for analysis.