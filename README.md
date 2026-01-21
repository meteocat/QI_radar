
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
├── 
├── config.txt                      # Configuration file with processing parameters
├── README.md                       # This documentation file
├── data/                           # Directory containing data files
│   └── raw/                        # Raw radar data files
│       └── ... (additional files)
└── visualization/                  # Directory for visualization outputs
```

---

## **Input Data Format**

### ▶ **Input Files**

The pipeline requires the following input data:

- **Raw Radar Data Files**: IRIS format (.RAW) files containing polar radar reflectivity data from the XRAD C-band radar network. These files are organized in subdirectories under `data/raw/`, named by radar station and date (e.g., `CDVRAW20250921/` for CDV radar on September 21, 2025). Each file corresponds to a volume scan at specific times.

- **Configuration File**: `config.txt` - A text file specifying processing parameters including:
  - Initial and final UTC times for processing
  - Volume scan type (VOLA, VOLB, or VOLBC)
  - CAPPI height in meters
  - Cartesian grid resolution
  - Paths to Digital Elevation Model (DEM) files for short-range and long-range processing
  - Path to echo tops climatology data (NetCDF file)

- **Auxiliary Data**:
  - DEM files (GeoTIFF format) for terrain correction
  - Climatological echo tops data (NetCDF format) for quality index computation

The raw radar data should be placed in the `data/raw/` directory with the expected naming convention and folder structure.

---

## **Output Products**

### ▶ **Output Files**

The pipeline generates the following products, saved as NetCDF (.nc) files in a structured directory hierarchy:

- **CAPPI (Constant Altitude Plan Position Indicator)**: Cartesian reflectivity fields at a constant altitude (specified in config.txt), with quality index and elevation data.

- **LUE (Lowest Usable Elevation)**: Products derived from the lowest usable elevation angles for each radar gate.

- **Composites**: Network-wide composites combining data from multiple radars using quality-based weighting (e.g., MAXQI method prioritizing highest quality index).

Each output file contains variables:
- `Z`: Reflectivity (dBZ)
- `QI`: Quality Index (dimensionless, 0-1)
- `RAD`: Radar identifier (integer)
- `ELEV`: Elevation angle (degrees)

Files are organized by volume type, product type (e.g., CAPPI, LUE), composite type, and date (YYYY/MM/DD). For example: `VOLB/CAPPI/MAXQI/2025/09/21/VOLB_CAPPI_MAXQI_2509211606.nc`

Visualization outputs (plots, maps) may be generated in the `visualization/` directory for analysis.

---

## **Getting Started**

### **1. Install Dependencies**

Install the required Python libraries using pip or conda:

```bash
pip install wradlib numpy scipy matplotlib cartopy pandas xarray xradar pyproj shapely rasterio
```

Or using conda:

```bash
conda install -c conda-forge wradlib numpy scipy matplotlib cartopy pandas xarray xradar pyproj shapely rasterio
```

### **2. Run the Main Pipeline**

Typical usage might look like:

```bash
python main/compute_qi.py
python main/generate_lue.py
python main/generate_cappi.py
python main/composite_products.py
python main/validation.py