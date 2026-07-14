import wradlib as wrl
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely import STRtree
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import label
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

def remove_small_specks(image, min_size=12):
    mask = image != -32

    labeled, num = label(mask)
    
    # Count pixels per label
    counts = np.bincount(labeled.ravel())
    
    # Create mask of valid components
    keep = counts >= min_size
    keep[0] = False  # background
    
    cleaned = image.copy()
    cleaned[~keep[labeled]] = -32
    
    return cleaned

def polar2cartesian(ds_input, dl):
    # ======================================== INPUT DATA ========================================
    
    sitecoords = (ds_input.longitude.values, ds_input.latitude.values, ds_input.altitude.values)
    range_res = np.unique(np.diff(ds_input.range.values))[0]  # range resolution (meters)
    nbins = len(ds_input.range) # number of range bins
    nrays = len(ds_input.azimuth) # number of rays

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
    # ds_QI = xr.Dataset(ds)
    ds_input = ds_input.assign_coords(azimuth=np.arange(0, len(ds_input.azimuth.values), 1, dtype=int))
    # ds_QI["QI"] = (("azimuth", "range"), QI)

    # ================================ NEAR-FIELD ALGORITHM ================================
    
    # Get radar points and values in UTM coordinates near-field
    ds_nearField = ds_input.sel(range=slice(0,D+dl/2)) # From 0 to D+dl/2 meters from the radar
    swp = ds_nearField.wrl.georef.georeference()
    proj_utm = wrl.georef.epsg_to_osr(25831)
    polygons = swp.wrl.georef.spherical_to_polyvert(crs=proj_utm, keep_attrs=True).values
    centroids = swp.wrl.georef.spherical_to_centroids(crs=proj_utm, keep_attrs=True).values
    x, y, z = centroids[..., 0], centroids[..., 1], centroids[..., 2]
    polar_points = np.array([x.ravel(), y.ravel()]).transpose()
    polar_values = ds_nearField.Z.values.ravel()
    QI_polar_values = ds_nearField.Q.values.ravel()

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
    ds_farField = ds_input.sel(range=slice(D-dl/2,None)) # From D-dl/2 to grid limit in meters from the radar
    swp = ds_farField.wrl.georef.georeference()
    proj_utm = wrl.georef.epsg_to_osr(25831)
    centroids = swp.wrl.georef.spherical_to_centroids(crs=proj_utm, keep_attrs=True).values
    x, y, z = centroids[..., 0], centroids[..., 1], centroids[..., 2]
    polar_points = np.array([x.ravel(), y.ravel()]).transpose()
    polar_values = ds_farField.Z.values.ravel()
    QI_polar_values = ds_farField.Q.values.ravel()

    # Find radar site coordinates in UTM
    x_center, y_center = to_utm.transform(sitecoords[0], sitecoords[1])

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

    # Distingwish when all gates are NaN
    all_nan = np.all(Z_values==-32, axis=1)

    # Define corrected Z method depending on sum_QI
    Z_corr_eq0 = np.sum(Z_values * weights, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_corr_gt0 = np.sum(Z_values * weights * QI_values, axis=1) / sum_QI

    # Apply Z correction depending on sum_QI and on nan condition
    flat_Z_corr = np.zeros_like(sum_QI)
    flat_Z_corr[sum_QI > 0] = Z_corr_gt0[sum_QI > 0]
    flat_Z_corr[sum_QI == 0] = Z_corr_eq0[sum_QI == 0]
    flat_Z_corr[all_nan] = -32
    
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
    swp = ds_input.wrl.georef.georeference()
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

    # ================================= REMOVE DETECTED SPECKS ================================
    Z_PPI_cart = remove_small_specks(Z_PPI_cart, min_size=12)

    return {
        "Z": Z_PPI_cart,
        "QI": QI_PPI_cart,
        "H": altitudes,
        "xgrid": xgrid,
        "ygrid": ygrid,
    }