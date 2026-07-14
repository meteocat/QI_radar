import wradlib as wrl
import numpy as np
import datetime as dt
import logging
import sys
import xarray as xr
xr.set_options(use_new_combine_kwarg_defaults=True)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

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

    try:
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
        logger.debug(f"Processor data extracted successfully: zcal={data['zcal']}, bw_h={data['bw_h']}")
        return data
    except Exception as e:
        logger.error(f"Error extracting processor data: {e}")
        raise

def single_PPI(ds, TOP12_clim_path, DEM_values, DEM_coords, instr_var):
    ''' Process all quality indices for a single PPI radar sweep and transform it to Cartesian coordinates.

    :param ds: xarray Dataset with radar polar data
    :param TOP12_clim_path: Path to the TOP12 climatology file
    :param DEM_values: DEM raster values
    :param DEM_coords: DEM raster coordinates
    :param instr_var: Dictionary with radar calibration and processing constants
    
    :return: 2D arrays with Cartesian reflectivity and Quality Index
    '''

    # ======================================== RADAR DATA ========================================

    # specify radar settings automatically
    sitecoords = (ds.longitude.values, ds.latitude.values, ds.altitude.values)
    nrays = len(ds.azimuth) # number of rays
    nbins = len(ds.range) # number of range bins
    el = ds.sweep_fixed_angle.values  # vertical antenna pointing angle (deg)
    range_res = np.unique(np.diff(ds.range.values))[0]  # range resolution (meters)
    logger.debug(f"PPI settings - elevation: {el:.2f}°, nrays: {nrays}, nbins: {nbins}, range_res: {range_res}m")
   
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

    QCL[QCL > 1] = 1
    QCL[QCL < 0] = 0
    QFI[QFI > 1] = 1
    QFI[QFI < 0] = 0

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

    QOS[QOS > 1] = 1
    QOS[QOS < 0] = 0

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

    QDeltaZ[QDeltaZ > 1] = 1
    QDeltaZ[QDeltaZ < 0] = 0

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

    QMDR[QMDR > 1] = 1
    QMDR[QMDR < 0] = 0

    # ================================== TOTAL QUALITY INDEX ==================================

    # QI for echoes detected and undetected. Note that quality in detected regions lacks 
    # the height quality index, which will be applied in the product generation step.
    QDET = (QCL * QDeltaZ)**(1/3)
    QUNDET = QOS * QFI * QMDR

    QDET[QDET > 1] = 1
    QDET[QDET < 0] = 0
    QUNDET[QUNDET > 1] = 1
    QUNDET[QUNDET < 0] = 0

    # Combine both QIs into a single array
    Z = ds.DBZH.values
    QI = np.copy(QDET)
    QI[Z == -32] = QUNDET[Z == -32]

    # Crop Quality Index so it fits the 0 to 1 margin
    QI[QI > 1] = 1
    QI[QI < 0] = 0
    
    return {
        "Q": QI,
        "QCL": QCL,
        "QDeltaZ": QDeltaZ,
        "QOS": QOS,
        "QFI": QFI,
        "QMDR": QMDR,
        "QDET": QDET,
        "QNDET": QUNDET,
    }