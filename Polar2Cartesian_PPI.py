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

def _read_int(n, signed=False, ang=False):
    
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
 
    f = 1
    if ang:
        f = 360/2**(n*8)
    return lambda s, by: int.from_bytes(by[s:s+n], byteorder=sys.byteorder, signed=signed)*f


def get_processor_data(IRIS_path):
    # Finds radar calibration constant
    hdr_rec_num = 2  # Number of header records
    rec_size = 6144  # Byte size of record

    with open(IRIS_path, "rb") as conn_in:
        data_all = bytearray(conn_in.read())

    data_hdr= data_all[0:hdr_rec_num*rec_size]

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


def single_PPI(ds, TOP12_clim_path, DEM_values, DEM_coords, instr_var, dl=1000, timings=False):

    # =========================== RADAR DATA ===========================

    if timings:
        t0 = time()

    # specify radar settings automatically
    sitecoords = (ds.longitude.values, ds.latitude.values, ds.altitude.values)
    nrays = len(ds.azimuth) # number of rays
    nbins = len(ds.range) # number of range bins
    el = ds.sweep_fixed_angle.values  # vertical antenna pointing angle (deg)
    range_res = np.unique(np.diff(ds.range.values))[0]  # range resolution (meters)

    if timings:
        t1 = time()
        print(f"RADAR DATA: {np.round((t1-t0),2)}s")
   
    # =========================== NON METEOROLOGICAL ECHOES ===========================

    Z = ds.DBZH.values
    T = ds.DBTH.values
    reg_SNR = (T == -32) * (Z == -32)
    reg_CCOR_SQI = (T != -32) * (Z == -32)
    reg_CL = (T != -32) * (Z != -32)

    QFI = np.ones_like(Z)
    QCL = np.ones_like(Z)

    QFI[reg_SNR] = 1
    QFI[reg_CCOR_SQI] = 0

    QCL[reg_CL] = 10**((Z[reg_CL] - T[reg_CL]) / 10)

    if timings:
        t2 = time()
        print(f"NO-MET DATA: {np.round((t2-t1),2)}s")

    # =========================== BEAM BLOCKAGE COMPUTATION (PBB) ===========================

    # Get range, beam radius and elevation grids
    # az = np.deg2rad(ds.azimuth.values)
    ra = ds.range.values
    beamradius = wrl.util.half_power_radius(ra, instr_var["bw_h"])
    r, elev = np.meshgrid(ra, ds.elevation.values)

    # Calculate the spherical coordinates of the bin centroids and their longitude, latitude and altitude
    coord = wrl.georef.sweep_centroids(nrays, range_res, nbins, el)
    coords = wrl.georef.spherical_to_proj(
        coord[..., 0], coord[..., 1], coord[..., 2], sitecoords
    )
    # lon = coords[..., 0]
    # lat = coords[..., 1]
    alt = coords[..., 2]
    polcoords = coords[..., :2]

    # Compute radar coordinate limits
    # rlimits = (lon.min(), lat.min(), lon.max(), lat.max())

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

    # Partial beam blockage QI
    # QPBB = np.zeros_like(CBB)
    # QPBB[CBB <= 0.9] = 1 - CBB[CBB <= 0.9]

    if timings:
        t3 = time()
        print(f"BBK QI: {np.round((t3-t2),2)}s")

    # =========================== RANGE QUALITY INDEX ===========================

    # Compute height, Beam Height QI and Range QI
    # D = 120000
    # QD = np.exp(-r**2/D**2)

    # if timings:
    #     t4 = time()
    #     print(f"R QI: {np.round((t4-t3),2)}s")

    # =========================== OVERSHOOTING QUALITY INDEX ===========================

    time_str = str(ds.time.values[0])[:16]
    month = dt.datetime.strptime(time_str, '%Y-%m-%dT%H:%M').month
    beam_h = alt/1000

    ds_clim_TOPS = xr.open_dataset(TOP12_clim_path, engine='scipy')
    heights = ds_clim_TOPS.height.values
    hist_values = ds_clim_TOPS.TOP12_HIST.sel(month=month).values
    cumsum = np.cumsum(hist_values)
    cumsum = cumsum/np.max(cumsum)

    Q2 = heights[np.abs(cumsum - 0.5).argmin()]
    Q3 = heights[np.abs(cumsum - 0.75).argmin()]

    mean = Q2
    std = (Q3 - Q2) / 0.67

    QOS = 1/2 + (mean - beam_h) / (np.sqrt(2*np.pi) * std)
    h0, h1 = mean - np.sqrt(np.pi/2)*std, mean + np.sqrt(np.pi/2)*std
    QOS[beam_h <= h0] = 1
    QOS[beam_h >= h1] = 0

    if timings:
        ti = time()
        print(f"H QI: {np.round((ti-t3),2)}s")

    # =========================== ATTENUATION COMPUTATION (PIA) ===========================

    # HARRISON ET. AL. (2000)
    PIA = wrl.atten.correct_attenuation_hb(
        ds.DBZH, coefficients=dict(a=4.57e-5, b=0.731, gate_length=1.0), mode="nan", thrs=59.0
    )
    PIA[PIA > 4.8] = 4.8

    # QA = 1 - (PIA - 1) / (10 - 1)
    # QA[PIA <= 1] = 1
    # QA[PIA >= 10] = 0
    # QA[np.isnan(PIA)] = 0

    if timings:
        t5 = time()
        print(f"A QI: {np.round((t5-ti),2)}s")

    # =========================== REFLECTIVITY ACCURACY (∆Z) QI ===========================
    Omega = np.deg2rad(instr_var["bw_h"]) # radians
    DivZ = 6 # dB/km

    PBB_DeltaZ = 10 * np.log10(1-CBB+1e-5)
    NonUnif_DeltaZ = 0.01 * Omega**2 * DivZ**2 * (r/1000)**2
    AbsDeltaZ = np.abs(PBB_DeltaZ + NonUnif_DeltaZ - PIA)

    QDeltaZ = (10 - AbsDeltaZ) / (10 - 1)
    QDeltaZ[AbsDeltaZ <= 1] = 1
    QDeltaZ[AbsDeltaZ >= 10] = 0
    QDeltaZ[CBB == 1] = 0

    if timings:
        t4 = time()
        print(f"DeltaZ QI: {np.round((t4-t5),2)}s")

    # =========================== MINIMUM DETECTABLE REFLECTIVITY QI ===========================
    PBB_fact = -10*np.log10(1-CBB+1e-5)
    MDR = instr_var["zcal"] + instr_var["SNR_th"] + 20 * np.log10(r / 1000) + \
          instr_var["gas_at"] * 1e-3 * r + PIA + PBB_fact

    MDR_min, MDR_max = 7, 15
    QMDR = (MDR_max - MDR) / (MDR_max - MDR_min)
    QMDR[MDR <= MDR_min] = 1
    QMDR[MDR >= MDR_max] = 0

    if timings:
        t6 = time()
        print(f"MDR QI: {np.round((t6-t5),2)}s")

    # =========================== TOTAL QUALITY INDEX ===========================

    # QI echoes detected and undetected
    QDET = (QCL * QDeltaZ)**(1/3)
    QUNDET = QOS * QFI * QMDR

    Z = ds.DBZH.values
    QI = np.copy(QDET)
    QI[Z == -32] = QUNDET[Z == -32]

    QI[QI > 1] = 1
    QI[QI < 0] = 0

    if timings:
        t7 = time()
        print(f"TOTAL QI: {np.round((t7-t6),2)}s")

    # =========================== TRANSFORMATION PRELIMINARIES ===========================

    # Define 2D cartesian grid characteristics
    if nbins < 200: # SHORT-RANGE
        lon_min, lon_max = -0.63, 4.58
        lat_min, lat_max = 39.89, 43.04
    else: # LONG-RANGE
        lon_min, lon_max = -2.01, 6.10
        lat_min, lat_max = 38.76, 44.09

    d_az = 360 / nrays # step in azimuth (deg)
    D = np.sqrt((9500*(1.3/d_az + 2300/range_res + 1.6*dl/1000)-39000)/np.pi) * 1000 # Near-field threshold (m)

    # CREATE CARTESIAN 2D GRID
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
    x_min, y_min = to_utm.transform(lon_min, lat_min)
    x_max, y_max = to_utm.transform(lon_max, lat_max)
    x0, x1 = sorted([x_min, x_max])
    y0, y1 = sorted([y_min, y_max])
    xgrid = np.arange(x0 + dl/2, x1, dl)
    ygrid = np.arange(y0 + dl/2, y1, dl)[::-1]
    grid_xy = np.meshgrid(xgrid, ygrid)

    # Include Quality-Index to the polar dataset
    ds_QI = xr.Dataset(ds)
    ds_QI = ds_QI.assign_coords(azimuth=np.arange(0, len(ds.azimuth.values), 1, dtype=int))
    ds_QI["QI"] = (("azimuth", "range"), QI)

    if timings:
        t8 = time()
        print(f"TRANSF. PRELIM.: {np.round((t8-t7),2)}s")

    # =========================== NEAR-FIELD ALGORITHM ===========================

    # GET RADAR POINTS AND VALUES IN UTM COORDINATES NEAR-FIELD
    ds_nearField = ds_QI.sel(range=slice(0,D+dl/2))
    swp = ds_nearField.wrl.georef.georeference()
    proj_utm = wrl.georef.epsg_to_osr(25831)
    polygons = swp.wrl.georef.spherical_to_polyvert(crs=proj_utm, keep_attrs=True).values
    centroids = swp.wrl.georef.spherical_to_centroids(crs=proj_utm, keep_attrs=True).values
    x, y, z = centroids[..., 0], centroids[..., 1], centroids[..., 2]
    polar_points = np.array([x.ravel(), y.ravel()]).transpose()
    polar_values = ds_nearField.DBZH.values.ravel()
    QI_polar_values = ds_nearField.QI.values.ravel()

    if timings:
        t81 = time()
        print(f"\tCentroids: {np.round((t81-t8),2)}s")

    # RESIZE GRIDS SO TO ONLY AFFECT NEAR-FIELD
    xPol_min, xPol_max = polar_points[:,0].min(), polar_points[:,0].max()
    yPol_min, yPol_max = polar_points[:,1].min(), polar_points[:,1].max()
    xgrid_near = xgrid[(xPol_min < xgrid)*(xgrid < xPol_max)]
    ygrid_near = ygrid[(yPol_min < ygrid)*(ygrid < yPol_max)]

    if timings:
        t82 = time()
        print(f"\tResize: {np.round((t82-t81),2)}s")

    # Create Polygon objects (shapely)
    polygons_list = [Polygon(coords) for coords in polygons]

    # Create Spatial Index (R-tree)
    tree = STRtree(polygons_list)

    if timings:
        t83 = time()
        print(f"\tCreate polygons & tree: {np.round((t83-t82),2)}s")

    # Generate cells using box (shapely)
    cells, cell_indices = [], []
    for i, x in enumerate(xgrid_near):
        for j, y in enumerate(ygrid_near):
            xmin, xmax = x - dl/2, x + dl/2
            ymin, ymax = y - dl/2, y + dl/2
            cells.append(box(xmin, ymin, xmax, ymax))
            cell_indices.append((i, j))

    if timings:
        t84 = time()
        print(f"\tGenerate cells: {np.round((t84-t83),2)}s")

    # Find intersections and compute corrected Z
    Z_corr_nearField = np.ones_like(grid_xy[0], dtype=float) * np.nan
    QI_corr_nearField = np.ones_like(grid_xy[0], dtype=float) * np.nan
    # cart_data_inters = np.ones_like(grid_xy[0], dtype=float) * np.nan
    for idx, cell in zip(cell_indices, cells):
        # Find polygon candidates indexs
        candidate_idxs = tree.query(cell, predicate=None)  # only bounding boxes
        # Filter only those which intersect
        intersecting_ids = [
            i for i in candidate_idxs if polygons_list[i].intersects(cell)
        ]
        if intersecting_ids:
            # Compute corrected Z from quality indeces
            Z_values = polar_values[np.array(intersecting_ids)]
            QI_values = QI_polar_values[np.array(intersecting_ids)]
            sum_QI = np.sum(QI_values)

            if sum_QI > 0:
                corr_Z = np.sum(Z_values * QI_values) / sum_QI
            else:
                corr_Z = np.nanmean(Z_values)

            corr_QI = np.nanmean(QI_values)

            Z_corr_nearField[ygrid==ygrid_near[idx[1]], xgrid==xgrid_near[idx[0]]] = corr_Z
            QI_corr_nearField[ygrid==ygrid_near[idx[1]], xgrid==xgrid_near[idx[0]]] = corr_QI
            # cart_data_inters[ygrid==ygrid_near[idx[1]], xgrid==xgrid_near[idx[0]]] = len(np.array(intersecting_ids))

    if timings:
        t85 = time()
        print(f"\tFind intersections: {np.round((t85-t84),2)}s")

        t9 = time()
        print(f"NEAR-FIELD: {np.round((t9-t8),2)}s")

    # =========================== FAR-FIELD ALGORITHM ===========================

    # GET RADAR POINTS AND VALUES IN UTM COORDINATES FAR-FIELD
    ds_farField = ds_QI.sel(range=slice(D-dl/2,None))
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

    # Inverse distance technique
    w = 1 / distances**2
    we = np.zeros((len(w[:,0]), 4))
    for i in range(4): we[:,i] = np.sum(w, axis=1)
    weights = w/we

    # Compute weighted mean
    Z_values = polar_values[indices]
    QI_values = QI_polar_values[indices]
    sum_QI = np.sum(weights * QI_values, axis=1)

    Z_corr_eq0 = np.sum(Z_values * weights, axis=1)
    Z_corr_gt0 = np.sum(Z_values * weights * QI_values, axis=1) / sum_QI

    flat_Z_corr = np.zeros_like(sum_QI)
    flat_Z_corr[sum_QI > 0] = Z_corr_gt0[sum_QI > 0]
    flat_Z_corr[sum_QI == 0] = Z_corr_eq0[sum_QI == 0]
    
    flat_QI_corr = np.sum(weights * QI_values, axis=1)

    # Assign values to 2D grid
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

    if timings:
        t0 = time()
        print(f"FAR-FIELD: {np.round((t0-t9),2)}s")

    # =========================== COMNBINE NEAR AND FAR FIELD ===========================

    reg_near = QI_corr_nearField >= QI_corr_farField
    reg_far = QI_corr_nearField < QI_corr_farField

    Z_PPI_cart = np.fmax(Z_corr_nearField, Z_corr_farField)
    Z_PPI_cart[reg_near] = Z_corr_nearField[reg_near]
    Z_PPI_cart[reg_far] = Z_corr_farField[reg_far]

    QI_PPI_cart = np.fmax(QI_corr_nearField, QI_corr_farField)
    QI_PPI_cart[reg_near] = QI_corr_nearField[reg_near]
    QI_PPI_cart[reg_far] = QI_corr_farField[reg_far]

    intersec = (np.isnan(Z_corr_nearField)==0) * (np.isnan(Z_corr_farField)==0)

    # Crop Quality Index so it fits the 0 to 1 margin
    QI_PPI_cart[QI_PPI_cart > 1] = 1
    QI_PPI_cart[QI_PPI_cart < 0] = 0
    QI_PPI_cart[np.isnan(QI_PPI_cart)*(np.isnan(Z_PPI_cart)==0)] = 0

    if timings:
        t1 = time()
        print(f"COMBINE FIELDS: {np.round((t1-t0),2)}s")

    # =========================== CREATE HEIGHT RASTER IN 2D CARTESIAN ===========================

    swp = ds_QI.wrl.georef.georeference()
    proj_utm = wrl.georef.epsg_to_osr(25831)
    centroids = swp.wrl.georef.spherical_to_centroids(crs=proj_utm, keep_attrs=True).values
    x, y, z = centroids[..., 0], centroids[..., 1], centroids[..., 2]

    # Flatten your original arrays
    points = np.column_stack((x.ravel(), y.ravel()))  # shape (N, 2)
    values = z.ravel()                                # shape (N,)

    # Create the meshgrid of target centers
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    # Interpolate
    altitudes = griddata(points, values, (Xgrid, Ygrid), method='nearest')

    # polar_points = np.array([x.ravel(), y.ravel()]).transpose()

    # # Build KD-tree from centroid coordinates
    # centroids_tree = cKDTree(polar_points)  # centroids shape (N, 2)

    # # Generate all grid cell centers as coordinate pairs
    # # assuming xgrid_far and ygrid_far are 1D arrays of cell centers
    # xg, yg = np.meshgrid(xgrid, ygrid, indexing='ij')
    # cell_centers = np.column_stack([xg.ravel(), yg.ravel()])  # shape (M, 2)

    # # Query the nearest centroid for each cell
    # distances, indices = centroids_tree.query(cell_centers, k=1)

    # # Save a height raster
    # flat_height = alt.ravel()[indices]
    # altitudes = np.ones_like(grid_xy[0], dtype=float) * np.nan
    # i = 0
    # for xcell, ycell in cell_centers:
    #     center_dist = np.sqrt((x_center-xcell)**2+(y_center-ycell)**2)
    #     if center_dist < ra[-1]:
    #         altitudes[ygrid==ycell, xgrid==xcell] = flat_height[i]
    #     i += 1
    
    if timings:
        t2 = time()
        print(f"HEIGHT RASTER: {np.round((t2-t1),2)}s")
    

    return Z_PPI_cart, QI_PPI_cart, altitudes, xgrid, ygrid


def Polar2Cartesian(IRIS_path, TOP12_clim_path, DEM_values, DEM_coords, dl=1000, save_dir="", sweeps=[]):
    # MANUAL DEFINITIONS
    # IRIS_path = "/home/nvm/nvm/data/rad_data/PBERAW20230727/PBE230727203024.RAWXBG5" # IRIS PBE 27/07/2023 20:30:24 PPI VOL-B
    # DEM_path = "/home/nvm/nvm/data/DEM/dem_cat_utm100_wgs84.tif"
    # bw = 1.0  # half power beam width (deg)

    file_name = os.path.splitext(os.path.basename(IRIS_path))[0]
    instr_var = get_processor_data(IRIS_path)

    # Import radar data
    vol = xd.io.open_iris_datatree(IRIS_path, reindex_angle=False)

    # Compute PPIs for each sweep
    Z_data, QI_data, H_data = [], [], []

    if sweeps == []:
        N_sweeps = len(vol["sweep_fixed_angle"].values)
        sweep_list = np.arange(N_sweeps)
    else:
        sweep_list = np.array(sweeps)
        N_sweeps = len(sweep_list)

    print(f"Computing PPIs {file_name} (0/{N_sweeps})", end='\r')
    for sweep in sweep_list: # VOL-B has 7 sweeps (0 to 6)
        ds = vol[f"sweep_{sweep}"]

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

    result.to_netcdf(f"{save_dir}/{file_name}.nc", engine="scipy")
    print()

    return result