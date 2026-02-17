import xradar as xd
import wradlib as wrl
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely import STRtree
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import datetime as dt
import os, sys
import xarray as xr
xr.set_options(use_new_combine_kwarg_defaults=True)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from time import time

def _read_int(n: int, signed=False, ang=False):
    """
    Reads a number of bytes in a byte stream,
    starting from a specified position.
    
    Parameters
    ----------
    n : int
        Number of bytes to pack.
    (by) : byte array
        Byte sequence from which to read.
    (s) : int
        Position in byte sequence from which the reading starts.
        
    Other parameters
    ----------------
    signed : bool
        Whether bytes should be read with signature.
    ang : bool
        Whether the quantity to be read is a binary angle
        (in IRIS nomenclature).
    
    Returns
    -------
    A function o the 'by' and 's' parameters that in turn returns:
    uintN/sintN : int
        Signed or unsigned integer/angular integer.   
    """
    
    # By default, no scaling factor
    f = 1

    # For angles, convert to degrees
    if ang:
        f = 360/2**(n*8)
    
    # Return a function that reads the bytes
    return lambda s, by: int.from_bytes(by[s:s+n], byteorder=sys.byteorder, signed=signed)*f


def get_processor_data(IRIS_path: str) -> dict:
    ''' Extract radar calibration and processing constants from IRIS raw file header.
    
    :param IRIS_path: Path to the raw IRIS radar file

    :return: Dictionary with radar calibration and processing constants
    '''

    # Finds radar calibration constant
    hdr_rec_num = 2  # Number of header records
    rec_size = 6144  # Byte size of record

    # Read raw file
    with open(IRIS_path, "rb") as conn_in:
        data_all = bytearray(conn_in.read())
    data_hdr= data_all[0:hdr_rec_num*rec_size]

    # Extract relevant parameters
    data = {
        "zcal": _read_int(2, signed=True)(7106, data_hdr) / 16,
        "bw_h": _read_int(4, signed=False, ang=True)(7952, data_hdr),
        "bw_v": _read_int(4, signed=False, ang=True)(7956, data_hdr),
        "gas_at": _read_int(2, signed=False)(6938, data_hdr) / 100000,
        "SNR_th": _read_int(2, signed=True)(7090, data_hdr) / 16,
        "CCOR_th": _read_int(2, signed=True)(7092, data_hdr) / 16,
        "SQI_th": _read_int(2, signed=True)(7094, data_hdr) / 256,
        "POW_th": _read_int(2, signed=True)(7096, data_hdr) / 256,
    }

    return data


def single_PPI(ds, TOP12_clim_path, DEM_values, DEM_coords, instr_var, dl=1000):
    ''' Process all quality indices for a single PPI radar sweep and transform it to Cartesian coordinates.

    :param ds: xarray Dataset with radar polar data
    :param TOP12_clim_path: Path to the TOP12 climatology file
    :param DEM_values: DEM raster values
    :param DEM_coords: DEM raster coordinates
    :param instr_var: Dictionary with radar calibration and processing constants
    :param dl: Cartesian grid resolution (meters), default set to 1000 m
    :param timings: Whether to print timing information
    
    :return: 2D arrays with Cartesian reflectivity and Quality Index
    '''

    # ======================================== RADAR DATA ========================================

    # specify radar settings automatically
    sitecoords = (ds.longitude.values, ds.latitude.values, ds.altitude.values)
    nrays = len(ds.azimuth) # number of rays
    nbins = len(ds.range) # number of range bins
    el = ds.sweep_fixed_angle.values  # vertical antenna pointing angle (deg)
    range_res = np.unique(np.diff(ds.range.values))[0]  # range resolution (meters)
   
    # ====== DATA FILTERING IN THE PROCESSOR (FI) AND GROUND CLUTTER ECHO CORRECTION (CL) ======

    # Define reflectivity arrays before (T) and after (Z) filtering
    Z = ds.DBZH.values
    T = ds.DBTH.values

    reg_SNR = (T == -32) * (Z == -32)       # Region affected by SNR filtering in T
    reg_CCOR_SQI = (T != -32) * (Z == -32)  # Region affected by Z filtering
    reg_CL = (T != -32) * (Z != -32)        # Region not filtered but altered by clutter correction

    # Initialize Quality Indices arrays
    QFI = np.ones_like(Z)
    QCL = np.ones_like(Z)

    # Apply QFI conditions
    QFI[reg_SNR] = 1
    QFI[reg_CCOR_SQI] = 0

    # Apply QCL computation
    QCL[reg_CL] = 10**((Z[reg_CL] - T[reg_CL]) / 10)

    # =========================== BEAM BLOCKAGE COMPUTATION (PBB) ===========================

    # Get range, beam radius and elevation grids
    ra = ds.range.values
    beamradius = wrl.util.half_power_radius(ra, instr_var["bw_h"])
    r, elev = np.meshgrid(ra, ds.elevation.values)

    # Calculate the spherical coordinates of the bin centroids and their altitude
    coord = wrl.georef.sweep_centroids(nrays, range_res, nbins, el)
    coords = wrl.georef.spherical_to_proj(
        coord[..., 0], coord[..., 1], coord[..., 2], sitecoords
    )
    alt = coords[..., 2]
    polcoords = coords[..., :2]

    # Map DEM rastervalues to polar grid points
    DEM_polarvalues = wrl.ipol.cart_to_irregular_spline(
        DEM_coords, DEM_values, polcoords, order=3, prefilter=False
    )

    # Calculate Beam Blockage
    np.seterr(invalid='ignore')
    PBB = wrl.qual.beam_block_frac(DEM_polarvalues, alt, beamradius)
    PBB = np.ma.masked_invalid(PBB)

    # Cumulative beam blockage
    CBB = wrl.qual.cum_beam_block_frac(PBB)

    # =========================== OVERSHOOTING QUALITY INDEX (OS) ===========================
    
    # Get month from time stamp
    time_str = str(ds.time.values[0])[:16]
    month = dt.datetime.strptime(time_str, '%Y-%m-%dT%H:%M').month

    # Compute beam height above sea level in km
    beam_h = alt/1000

    # Open TOP12dBZ climatology file and extract percentiles p50 and p75
    # for the current month
    ds_clim_TOPS = xr.open_dataset(TOP12_clim_path, engine='scipy')
    heights = ds_clim_TOPS.height.values
    hist_values = ds_clim_TOPS.TOP12_HIST.sel(month=month).values
    cumsum = np.cumsum(hist_values)
    cumsum = cumsum/np.max(cumsum)
    Q2 = heights[np.abs(cumsum - 0.5).argmin()]
    Q3 = heights[np.abs(cumsum - 0.75).argmin()]

    # Find mean and standard deviation of the overshooting height distribution
    mean = Q2
    std = (Q3 - Q2) / 0.67

    # Compute Overshooting Quality Index
    QOS = 1/2 + (mean - beam_h) / (np.sqrt(2*np.pi) * std)
    h0, h1 = mean - np.sqrt(np.pi/2)*std, mean + np.sqrt(np.pi/2)*std
    QOS[beam_h <= h0] = 1
    QOS[beam_h >= h1] = 0

    # =========================== ATTENUATION COMPUTATION (PIA) ===========================

    # Use the HARRISON ET. AL. (2000) coefficients for attenuation correction
    PIA = wrl.atten.correct_attenuation_hb(
        ds.DBZH, coefficients=dict(a=4.57e-5, b=0.731, gate_length=1.0), mode="nan", thrs=59.0
    )
    PIA[PIA > 4.8] = 4.8 # Cap maximum PIA to 4.8 dB

    # =========================== REFLECTIVITY ACCURACY (∆Z) QI ===========================

    # Predefine necessary variables
    Omega = np.deg2rad(instr_var["bw_h"]) # radians
    DivZ = 6 # dB/km

    # Compute main factors involved in ∆Z calculation
    PBB_DeltaZ = 10 * np.log10(1-CBB+1e-5)                      # Beam Blockage
    NonUnif_DeltaZ = 0.01 * Omega**2 * DivZ**2 * (r/1000)**2    # Non-uniform beam filling
    AbsDeltaZ = np.abs(PBB_DeltaZ + NonUnif_DeltaZ - PIA)       # Total absolute error

    # Compute ∆Z Quality Index
    QDeltaZ = (10 - AbsDeltaZ) / (10 - 1)
    QDeltaZ[AbsDeltaZ <= 1] = 1
    QDeltaZ[AbsDeltaZ >= 10] = 0
    QDeltaZ[CBB == 1] = 0

    # =========================== MINIMUM DETECTABLE REFLECTIVITY QI ===========================

    # Compute Minimum Detectable Reflectivity
    PBB_fact = -10*np.log10(1-CBB+1e-5)             # Beam Blockage factor
    BeamBroad_fact = 20 * np.log10(r / 1000)        # Beam Broadening factor
    GasAtt_fact = instr_var["gas_at"] * 1e-3 * r    # Gaseous Attenuation factor
    MDR = instr_var["zcal"] + instr_var["SNR_th"] + BeamBroad_fact + GasAtt_fact + PIA + PBB_fact

    # Compute Minimum Detectable Reflectivity Quality Index
    MDR_min, MDR_max = 7, 15
    QMDR = (MDR_max - MDR) / (MDR_max - MDR_min)
    QMDR[MDR <= MDR_min] = 1
    QMDR[MDR >= MDR_max] = 0

    # ================================== TOTAL QUALITY INDEX ==================================

    # QI for echoes detected and undetected. Note that quality in detected regions lacks 
    # the height quality index, which will be applied in the product generation step.
    QDET = (QCL * QDeltaZ)**(1/3)
    QUNDET = QOS * QFI * QMDR

    # Combine both QIs into a single array
    Z = ds.DBZH.values
    QI = np.copy(QDET)
    QI[Z == -32] = QUNDET[Z == -32]

    # Crop Quality Index so it fits the 0 to 1 margin
    QI[QI > 1] = 1
    QI[QI < 0] = 0

    # ======================== CARTESIAN TRANSFORMATION PRELIMINARIES ========================

    # Define 2D cartesian grid dimensions depending on radar range
    if nbins < 200: # SHORT-RANGE
        lon_min, lon_max = -0.63, 4.58
        lat_min, lat_max = 39.89, 43.04
    else: # LONG-RANGE
        lon_min, lon_max = -2.01, 6.10
        lat_min, lat_max = 38.76, 44.09

    # Step in azimuth (deg)
    d_az = 360 / nrays 

    # Near-field / Far-field threshold limit (m)
    D = np.sqrt((9500*(1.3/d_az + 2300/range_res + 1.6*dl/1000)-39000)/np.pi) * 1000 

    # CREATE CARTESIAN 2D GRID
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
    x_min, y_min = to_utm.transform(lon_min, lat_min)
    x_max, y_max = to_utm.transform(lon_max, lat_max)
    x0, x1 = sorted([x_min, x_max])
    y0, y1 = sorted([y_min, y_max])
    xgrid = np.arange(x0 + dl/2, x1, dl)
    ygrid = np.arange(y0 + dl/2, y1, dl)[::-1]
    grid_xy = np.meshgrid(xgrid, ygrid)

    # Include the computed Quality-Index to the imported polar dataset
    ds_QI = xr.Dataset(ds)
    ds_QI = ds_QI.assign_coords(azimuth=np.arange(0, len(ds.azimuth.values), 1, dtype=int))
    ds_QI["QI"] = (("azimuth", "range"), QI)

    # ================================ NEAR-FIELD ALGORITHM ================================

    # Get radar points and values in UTM coordinates near-field
    ds_nearField = ds_QI.sel(range=slice(0,D+dl/2)) # From 0 to D+dl/2 meters from the radar
    swp = ds_nearField.wrl.georef.georeference()
    proj_utm = wrl.georef.epsg_to_osr(25831)
    polygons = swp.wrl.georef.spherical_to_polyvert(crs=proj_utm, keep_attrs=True).values
    centroids = swp.wrl.georef.spherical_to_centroids(crs=proj_utm, keep_attrs=True).values
    x, y, z = centroids[..., 0], centroids[..., 1], centroids[..., 2]
    polar_points = np.array([x.ravel(), y.ravel()]).transpose()
    polar_values = ds_nearField.DBZH.values.ravel()
    QI_polar_values = ds_nearField.QI.values.ravel()

    # Resize grids so to only affect near-field
    xPol_min, xPol_max = polar_points[:,0].min(), polar_points[:,0].max()
    yPol_min, yPol_max = polar_points[:,1].min(), polar_points[:,1].max()
    xgrid_near = xgrid[(xPol_min < xgrid)*(xgrid < xPol_max)]
    ygrid_near = ygrid[(yPol_min < ygrid)*(ygrid < yPol_max)]

    # Create Polygon objects (shapely)
    polygons_list = [Polygon(coords) for coords in polygons]

    # Create Spatial Index (R-tree)
    tree = STRtree(polygons_list)

    # Generate cells using box (shapely)
    cells, cell_indices = [], []
    for i, x in enumerate(xgrid_near):
        for j, y in enumerate(ygrid_near):
            xmin, xmax = x - dl/2, x + dl/2
            ymin, ymax = y - dl/2, y + dl/2
            cells.append(box(xmin, ymin, xmax, ymax))
            cell_indices.append((i, j))

    # Find intersections between polar-grid cells and cartesian-grid cells and compute corrected Z
    Z_corr_nearField = np.ones_like(grid_xy[0], dtype=float) * np.nan
    QI_corr_nearField = np.ones_like(grid_xy[0], dtype=float) * np.nan
    for idx, cell in zip(cell_indices, cells):
        # Find polygon candidates indexs
        candidate_idxs = tree.query(cell, predicate=None)  # only bounding boxes
        # Filter only those which intersect
        intersecting_ids = [i for i in candidate_idxs if polygons_list[i].intersects(cell)]
        if intersecting_ids:
            # Compute corrected Z from quality indeces
            Z_values = polar_values[np.array(intersecting_ids)]
            QI_values = QI_polar_values[np.array(intersecting_ids)]
            sum_QI = np.sum(QI_values)

            # Weighted mean and simple mean for reflectivity depending on QI sum
            if sum_QI > 0:
                corr_Z = np.sum(Z_values * QI_values) / sum_QI
            else:
                corr_Z = np.nanmean(Z_values)

            # Quality Index corrected as simple mean
            corr_QI = np.nanmean(QI_values)

            # Assign values to 2D grid
            Z_corr_nearField[ygrid==ygrid_near[idx[1]], xgrid==xgrid_near[idx[0]]] = corr_Z
            QI_corr_nearField[ygrid==ygrid_near[idx[1]], xgrid==xgrid_near[idx[0]]] = corr_QI

    # ================================ FAR-FIELD ALGORITHM ================================

    # Get radar points and values in UTM coordinates far-field
    ds_farField = ds_QI.sel(range=slice(D-dl/2,None)) # From D-dl/2 to grid limit in meters from the radar
    swp = ds_farField.wrl.georef.georeference()
    proj_utm = wrl.georef.epsg_to_osr(25831)
    centroids = swp.wrl.georef.spherical_to_centroids(crs=proj_utm, keep_attrs=True).values
    x, y, z = centroids[..., 0], centroids[..., 1], centroids[..., 2]
    polar_points = np.array([x.ravel(), y.ravel()]).transpose()
    polar_values = ds_farField.DBZH.values.ravel()
    QI_polar_values = ds_farField.QI.values.ravel()

    # Find radar site coordinates in UTM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
    x_center, y_center = transformer.transform(sitecoords[0], sitecoords[1])

    # Find far-field limits
    x_min_fF, x_max_fF = centroids[..., 0].min(), centroids[..., 0].max()
    y_min_fF, y_max_fF = centroids[..., 1].min(), centroids[..., 1].max()
    xgrid_far = xgrid[(x_min_fF < xgrid)*(xgrid < x_max_fF)]
    ygrid_far = ygrid[(y_min_fF < ygrid)*(ygrid < y_max_fF)]

    # Build KD-tree from centroid coordinates
    centroids_tree = cKDTree(polar_points)  # centroids shape (N, 2)

    # Generate all grid cell centers as coordinate pairs
    # assuming xgrid_far and ygrid_far are 1D arrays of cell centers
    xg, yg = np.meshgrid(xgrid_far, ygrid_far, indexing='ij')
    cell_centers = np.column_stack([xg.ravel(), yg.ravel()])  # shape (M, 2)

    # Query the 4 nearest centroids for each cell
    distances, indices = centroids_tree.query(cell_centers, k=4)

    # 'indices' has shape (n_cells, 4) with the indices of the 4 closest centroids
    # 'distances' has the corresponding distances

    # Inverse distance squared technique
    w = 1 / distances**2
    ws = np.zeros((len(w[:,0]), 4))
    for i in range(4): ws[:,i] = np.sum(w, axis=1)
    weights = w/ws

    # Compute weighted mean
    Z_values = polar_values[indices]
    QI_values = QI_polar_values[indices]
    sum_QI = np.sum(weights * QI_values, axis=1)

    # Define corrected Z method depending on sum_QI
    Z_corr_eq0 = np.sum(Z_values * weights, axis=1)
    Z_corr_gt0 = np.sum(Z_values * weights * QI_values, axis=1) / sum_QI

    # Apply Z correction depending on sum_QI
    flat_Z_corr = np.zeros_like(sum_QI)
    flat_Z_corr[sum_QI > 0] = Z_corr_gt0[sum_QI > 0]
    flat_Z_corr[sum_QI == 0] = Z_corr_eq0[sum_QI == 0]
    
    # Correct QI as weighted mean
    flat_QI_corr = np.sum(weights * QI_values, axis=1)

    # Assign values to 2D grid if it is within far-field limits
    Z_corr_farField = np.ones_like(grid_xy[0], dtype=float) * np.nan
    QI_corr_farField = np.ones_like(grid_xy[0], dtype=float) * np.nan
    inner_rad = np.abs(x_center - xPol_min)
    outer_rad = np.abs(x_center - x_min_fF)
    i = 0
    for xcell, ycell in cell_centers:
        center_dist = np.sqrt((x_center-xcell)**2+(y_center-ycell)**2)
        if inner_rad < center_dist and center_dist < outer_rad:
            Z_corr_farField[ygrid==ycell, xgrid==xcell] = flat_Z_corr[i]
            QI_corr_farField[ygrid==ycell, xgrid==xcell] = flat_QI_corr[i]
        i += 1

    # ================================ COMBINE NEAR AND FAR FIELD ================================

    # Define regions where each field has to be used by choosing the maximum QI value
    reg_near = QI_corr_nearField >= QI_corr_farField
    reg_far = QI_corr_nearField < QI_corr_farField

    # In the intersections, choose the values from the field with higher QI
    Z_PPI_cart = np.fmax(Z_corr_nearField, Z_corr_farField)
    Z_PPI_cart[reg_near] = Z_corr_nearField[reg_near]
    Z_PPI_cart[reg_far] = Z_corr_farField[reg_far]
    QI_PPI_cart = np.fmax(QI_corr_nearField, QI_corr_farField)
    QI_PPI_cart[reg_near] = QI_corr_nearField[reg_near]
    QI_PPI_cart[reg_far] = QI_corr_farField[reg_far]

    # Crop Quality Index so it fits the 0 to 1 margin
    QI_PPI_cart[QI_PPI_cart > 1] = 1
    QI_PPI_cart[QI_PPI_cart < 0] = 0

    # Set QI to 0 where there is quality data missing and reflectivity data available
    QI_PPI_cart[np.isnan(QI_PPI_cart)*(np.isnan(Z_PPI_cart)==0)] = 0

    # =========================== CREATE HEIGHT RASTER IN 2D CARTESIAN ===========================

    # Extract radar gate coordinates in UTM
    swp = ds_QI.wrl.georef.georeference()
    proj_utm = wrl.georef.epsg_to_osr(25831)
    centroids = swp.wrl.georef.spherical_to_centroids(crs=proj_utm, keep_attrs=True).values
    x, y, z = centroids[..., 0], centroids[..., 1], centroids[..., 2]

    # Flatten your original arrays
    points = np.column_stack((x.ravel(), y.ravel()))  # shape (N, 2)
    values = z.ravel()                                # shape (N,)

    # Create the meshgrid of target centers
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    # Interpolate to fit the cartesian grid
    altitudes = griddata(points, values, (Xgrid, Ygrid), method='nearest')

    return Z_PPI_cart, QI_PPI_cart, altitudes, xgrid, ygrid


def Polar2Cartesian(IRIS_path, TOP12_clim_path, DEM_values, DEM_coords, dl=1000, save_dir="", sweeps=[], VOLUME_NAME="unknown"):
    ''' Process all PPI radar sweeps in an IRIS raw file, compute the quality index and transform them to Cartesian coordinates.
    
    :param IRIS_path: Path to the raw IRIS radar file
    :param TOP12_clim_path: Path to the TOP12 climatology file
    :param DEM_values: DEM raster values
    :param DEM_coords: DEM raster coordinates
    :param dl: Cartesian grid resolution (meters), default set to 1000 m
    :param save_dir: Directory to save the output NetCDF file, default set to current directory
    :param sweeps: List of sweep indices to process, default set to all sweeps
    :param VOLUME_NAME: Volume name to be included in the output dataset attributes, default set to 'unknown'
    
    :return: xarray Dataset with Cartesian reflectivity and Quality Index for all sweeps
    '''

    # Extract file name without extension
    file_name = os.path.splitext(os.path.basename(IRIS_path))[0]

    # Extract radar calibration and processing constants
    instr_var = get_processor_data(IRIS_path)

    # Import radar data
    vol = xd.io.open_iris_datatree(IRIS_path, reindex_angle=False)

    # Initialize lists to store results and set sweep list
    Z_data, QI_data, H_data = [], [], []
    if sweeps == []:
        N_sweeps = len(vol["sweep_fixed_angle"].values)
        sweep_list = np.arange(N_sweeps)
    else:
        sweep_list = np.array(sweeps)
        N_sweeps = len(sweep_list)

    # Compute PPIs for each sweep
    print(f"Computing PPIs {file_name} (0/{N_sweeps})", end='\r')
    for sweep in sweep_list:
        # Extract sweep dataset
        ds = vol[f"sweep_{sweep}"]

        # Process single PPI and append results to lists
        Z_PPI_cart, QI_PPI_cart, altitudes, xgrid, ygrid = single_PPI(ds, TOP12_clim_path, 
                                                                      DEM_values, DEM_coords, 
                                                                      instr_var=instr_var, dl=dl)
        Z_data.append(Z_PPI_cart)
        QI_data.append(QI_PPI_cart)
        H_data.append(altitudes)

        print(f"Computing PPIs {file_name} ({sweep+1}/{N_sweeps})", end='\r')

    # Save to dataset
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
    x_loc, y_loc = to_utm.transform(vol.longitude.values, vol.latitude.values)
    result = xr.Dataset(
        {
            "Z": (["elev", "y", "x"], np.array(Z_data)),
            "QI": (["elev", "y", "x"], np.array(QI_data)),
            "H": (["elev", "y", "x"], np.array(H_data)),
            "x_loc": ([], x_loc),
            "y_loc": ([], y_loc),
            "z_loc": ([], vol.altitude.values),
        },
        coords={
            "elev": vol["sweep_fixed_angle"].values[sweep_list],
            "x": xgrid,
            "y": ygrid,
        },
    )
    result.attrs.update({
        "title": "Cartesian radar grid",
        "source": "Computed from polar radar volume",
        "volume_scan": VOLUME_NAME,
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    })

    result.to_netcdf(f"{save_dir}/{file_name}.nc", engine="scipy")
    print()

    return result