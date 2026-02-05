import numpy as np

def make_CAPPI(ds, CAPPI_H: int):
    ''' Create a CAPPI (Constant Altitude Plan Position Indicator) from an individual radar dataset.
    
    :param ds: xarray dataset containing reflectivity, quality and elevation data for each PPI and for a specific time step
    :param CAPPI_H: Desired height of the CAPPI in meters

    :return: CAPPI, QI, and ELEV arrays representing the reflectivity, quality index, and elevation used for each pixel in the CAPPI
    '''

    # Initialize CAPPI, QI and ELEV arrays with NaN values
    CAPPI = np.ones_like(ds.isel(elev=0).Z) * np.nan
    QI = np.ones_like(ds.isel(elev=0).Z) * np.nan
    ELEV = np.ones_like(ds.isel(elev=0).Z) * np.nan

    # Define a quality index threshold to determine if a pixel's value is reliable 
    # or if it should be replaced by values from other elevations
    QI_th = 0.1

    # ===================================== VALUES IN ZONE 1 =====================================
    # Select the highest elevation dataset
    e = len(ds.elev.values)-1
    ds_e = ds.isel(elev=e)

    # Extract reflectivity and quality index values for the highest elevation
    Z_e = ds_e.Z.values
    QI_e = ds_e.QI.values

    # Define region closer to radar by checking if height to ground is lower than CAPPI_H
    H_bot = ds_e.H.values
    reg = H_bot < CAPPI_H

    # Define CAPPI, QI and ELEV values in the region closer to radar with the highest elevation values
    CAPPI[reg] = Z_e[reg]
    QI[reg] = QI_e[reg]
    ELEV[reg] = e

    # Handle values projected with low quality, it projects from the beam below with available data
    while np.any(QI[reg] <= QI_th) and e > 0:
        LowQ_reg = reg * (QI <= QI_th) # Define low quality region

        e -= 1 # Move to the next elevation below
        
        # Select dataset for the new elevation and extract reflectivity and quality index values
        ds_e = ds.isel(elev=e)
        Z_e = ds_e.Z.values
        QI_e = ds_e.QI.values

        # Define new region closer to radar with the new elevation
        CAPPI[LowQ_reg] = Z_e[LowQ_reg]
        QI[LowQ_reg] = QI_e[LowQ_reg]
        ELEV[LowQ_reg] = e

    # ===================================== VALUES IN ZONE 2 =====================================
    # Iterating through beam elevations, from highest to lowest
    for e in range(len(ds.elev.values)-1, 0, -1):
        # Select upper and lower beams
        ds_top, ds_bot = ds.isel(elev=e), ds.isel(elev=e-1)
        Z_top, Z_bot = ds_top.Z.values, ds_bot.Z.values
        QI_top, QI_bot = ds_top.QI.values, ds_bot.QI.values
        
        # Find region within CAPPI altitude between upper and lower beams
        H_top, H_bot = ds_top.H.values, ds_bot.H.values
        reg = (H_top > CAPPI_H) * (H_bot < CAPPI_H)

        # Assign Z & QI to variables
        Z_top, Z_bot = Z_top[reg], Z_bot[reg]
        QI_top, QI_bot = QI_top[reg], QI_bot[reg]

        # Compute Z weighting with top and bottom values
        numerator = np.nansum([Z_top*QI_top, Z_bot*QI_bot], axis=0)
        denominator = np.nansum([QI_top, QI_bot], axis=0)
        denominator[denominator==0] = np.nan # Assign NaN values where QI = 0 in both top and bottom
        CAPPI_reg = numerator / denominator

        # Handle QI = 0 by doing mean
        num = np.nansum([Z_top, Z_bot], axis=0)
        CAPPI_reg[np.isnan(denominator)] = num[np.isnan(denominator)] / 2

        # Compute new QI by doing mean
        numerator = np.nansum([QI_top, QI_bot], axis=0)
        QI_reg =  numerator / 2

        # Assign to new array
        CAPPI[reg] = CAPPI_reg
        QI[reg] = QI_reg
        ELEV[reg] = ds.elev.values[e-1] # For elevation, assign the lower beam

        # Handle values with low quality, it projects from the pair of beams above
        # and below with available data
        e_top, e_bot = np.copy(e), np.copy(e-1)
        # It loops until there are no more low quality values in the region or until
        # it reaches the highest or lowest elevation available
        while np.any(QI[reg] <= QI_th) and (e_top < len(ds.elev.values)-1 or e_bot > 0):
            LowQ_reg = reg * (QI <= QI_th) # Define low quality region

            # Move to the next pair of beams above and below if available
            e_top += 1 if e_top < len(ds.elev.values)-1 else 0
            e_bot -= 1 if e_bot > 0 else 0
            
            # Repeat same process as before for each recalculated pair of elevations
            ds_top, ds_bot = ds.isel(elev=e_top), ds.isel(elev=e_bot)
            Z_top, Z_bot = ds_top.Z.values, ds_bot.Z.values
            QI_top, QI_bot = ds_top.QI.values, ds_bot.QI.values

            Z_top, Z_bot = ds_top.Z.values[LowQ_reg], ds_bot.Z.values[LowQ_reg]
            QI_top, QI_bot = QI_top[LowQ_reg], QI_bot[LowQ_reg]

            numerator = np.nansum([Z_top*QI_top, Z_bot*QI_bot], axis=0)
            denominator = np.nansum([QI_top, QI_bot], axis=0)
            denominator[denominator==0] = np.nan
            CAPPI_reg = numerator / denominator

            num = np.nansum([Z_top, Z_bot], axis=0)
            CAPPI_reg[np.isnan(denominator)] = num[np.isnan(denominator)] / 2

            numerator = np.nansum([QI_top, QI_bot], axis=0)
            QI_reg =  numerator / 2

            CAPPI[LowQ_reg] = CAPPI_reg
            QI[LowQ_reg] = QI_reg
            ELEV[LowQ_reg] = ds.elev.values[e_bot]

    # ===================================== VALUES IN ZONE 3 =====================================
    # For the region further from the radar, we will assign values from the highest
    # elevation with available data
    e = 0
    ds_e = ds.isel(elev=e)
    Z_e = ds_e.Z.values
    QI_e = ds_e.QI.values

    # Define region further from radar by checking if height to ground is higher than
    # CAPPI_H and there is data available
    H_top = ds_e.H.values
    reg = (H_top > CAPPI_H) * (np.isnan(ds_e.Z.values) == 0)

    # Set CAPPI, QI and ELEV values
    CAPPI[reg] = Z_e[reg]
    QI[reg] = QI_e[reg]
    ELEV[reg] = e

    # Handle values projected with low quality, it projects from the beam above with available data
    while np.any(QI[reg] <= QI_th) and e < len(ds.elev.values)-1:
        LowQ_reg = reg * (QI <= QI_th) # Define low quality region

        # Move to the next elevation above
        e += 1
        ds_e = ds.isel(elev=e)
        Z_e = ds_e.Z.values
        QI_e = ds_e.QI.values

        # Define new region where quality is imporved
        BetterQ_reg = LowQ_reg * (ds_e.QI.values > QI)

        # Set CAPPI, QI and ELEV values in the new region
        CAPPI[BetterQ_reg] = Z_e[BetterQ_reg]
        QI[BetterQ_reg] = QI_e[BetterQ_reg]
        ELEV[BetterQ_reg] = e

    return CAPPI, QI, ELEV


def make_LUE(ds, DEM_resampled):
    ''' Create a LUE (Lowest Usable Elevation) from an individual radar dataset by selecting the lowest elevation with good quality index for each pixel.
    
    :param ds: xarray dataset containing reflectivity, quality and elevation data for each PPI and for a specific time step
    :param DEM_resampled: 2D array of the DEM resampled to the radar grid, representing the height to ground for each pixel

    :return: LUE, QI, H and ELEV arrays representing the reflectivity, quality index, height to ground and elevation used for each pixel in the LUE
    '''

    # Define a quality index threshold to determine if a pixel's value is reliable 
    # or if it should be replaced by values from other elevations
    QI_th = 0.1

    # Extract lowest elevation variables (reflectivity, height to ground and quality index)
    LUE = ds.isel(elev=0).Z.values
    H = ds.isel(elev=0).H.values - DEM_resampled
    QI = ds.isel(elev=0).QI.values

    # Initialize the elevation array which will store what PPI elevation has been used
    ELEV = np.ones_like(LUE) * np.nan
    ELEV[np.isnan(LUE)==0] = ds.elev.values[0]

    # Iterate through all elevations until there is not a quality index lower than threshold or
    # until it reaches the highest elevation available
    e = 1 # Starts from the second elevation
    while (e < len(ds.elev.values)) and np.any(QI <= QI_th):
        # Assign new variables for current elevation
        Z_e = ds.isel(elev=e).Z.values
        QI_e = ds.isel(elev=e).QI.values
        H_e = ds.isel(elev=e).H.values - DEM_resampled

        # Pixels that meet the following conditions will be redefined:
        # 1. QI <= QI_th: The pixel has low quality index and needs to be redefined
        # 2. QI_e > QI_th: The above elevation has good quality index
        # 3. Z_e > LUE: The above pixel has higher reflectivity
        redef_reg = (QI <= QI_th) * (QI_e > QI_th) * (Z_e > LUE)

        # Pixels are redefined with the above elevation values
        ELEV[redef_reg] = ds.elev.values[e]
        H[redef_reg] = H_e[redef_reg]
        LUE[redef_reg] = Z_e[redef_reg]
        QI[redef_reg] = QI_e[redef_reg]

        e += 1

    return LUE, QI, H, ELEV