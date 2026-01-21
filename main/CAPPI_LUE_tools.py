import numpy as np

def distance_weighting(dist):
    H = 1000

    if H <= 0:
        return 0
    else:
        return (np.exp(-dist**2/H**2)) ** (1/3)


def make_CAPPI(ds, CAPPI_H):
    CAPPI = np.ones_like(ds.isel(elev=0).Z) * np.nan
    QI = np.ones_like(ds.isel(elev=0).Z) * np.nan
    ELEV = np.ones_like(ds.isel(elev=0).Z) * np.nan

    QI_th = 0.1

    # =========================== VALUES "INSIDE" HIGHEST BEAM ===========================
    e = len(ds.elev.values)-1
    # print(f'Inner layer: e={ds.elev.values[e]}')
    ds_e = ds.isel(elev=e)

    # region closer to radar
    H_bot = ds_e.H.values
    reg = H_bot < CAPPI_H

    # project Z & QI and adjust QI weighted by vertical distance
    # dis_bot = CAPPI_H - H_bot
    # w_dis_bot = distance_weighting(dis_bot)
    CAPPI[reg] = ds_e.Z.values[reg]
    QI[reg] = ds_e.QI.values[reg] # * w_dis_bot[reg]
    ELEV[reg] = e

    # handle NaN values not projected, it projects from the beam below with available data
    while np.any(QI[reg] <= QI_th) and e > 0:        
        NaN_reg = reg * (QI <= QI_th)

        e -= 1
        # print(f'Inner layer: e={ds.elev.values[e]}')
        ds_e = ds.isel(elev=e)

        H_bot = ds_e.H.values
        # dis_bot = CAPPI_H - H_bot
        # w_dis_bot = distance_weighting(dis_bot)
        CAPPI[NaN_reg] = ds_e.Z.values[NaN_reg]
        QI[NaN_reg] = ds_e.QI.values[NaN_reg] # * w_dis_bot[NaN_reg]
        ELEV[NaN_reg] = e

    # =========================== VALUES BETWEEN BEAMS ===========================
    # iterating through beams, from highest to lowest
    for e in range(len(ds.elev.values)-1, 0, -1):
        # print(f"\nInter layer: {ds.elev.values[e]}-{ds.elev.values[e-1]}")

        # Select upper and lower beams
        ds_top = ds.isel(elev=e)
        ds_bot = ds.isel(elev=e-1)
        
        # Find region within CAPPI altitude between upper and lower beams
        H_top, H_bot = ds_top.H.values, ds_bot.H.values
        reg = (H_top > CAPPI_H) * (H_bot < CAPPI_H)

        # Compute Z & QI in CAPPI level
        # 1. Assign Z & QI to variables
        Z_top, Z_bot = ds_top.Z.values[reg], ds_bot.Z.values[reg]
        QI_top, QI_bot = ds_top.QI.values[reg], ds_bot.QI.values[reg]

        # 2. Compute Z weighting with top and bottom values
        numerator = np.nansum([Z_top*QI_top, Z_bot*QI_bot], axis=0)
        denominator = np.nansum([QI_top, QI_bot], axis=0)
        denominator[denominator==0] = np.nan # Assign NaN values where QI = 0 in both top and bottom
        CAPPI_reg = numerator / denominator

        # 3. Handle QI = 0 by doing mean
        num = np.nansum([Z_top, Z_bot], axis=0)
        CAPPI_reg[np.isnan(denominator)] = num[np.isnan(denominator)] / 2

        # 4. Compute new QI by doing mean
        numerator = np.nansum([QI_top, QI_bot], axis=0)
        QI_reg =  numerator / 2

        # Assign to new array
        CAPPI[reg] = CAPPI_reg
        QI[reg] = QI_reg
        ELEV[reg] = ds.elev.values[e-1]

        # Handle QI == 0 values
        e_top, e_bot = np.copy(e), np.copy(e-1)
        while np.any(QI[reg] <= QI_th) and (e_top < len(ds.elev.values)-1 or e_bot > 0):
            NaN_reg = reg * (QI <= QI_th)

            e_top += 1 if e_top < len(ds.elev.values)-1 else 0
            e_bot -= 1 if e_bot > 0 else 0
            # print(f"\t{ds.elev.values[e_top]}-{ds.elev.values[e_bot]}")
            
            # Repeat same process as before for each recalculated pair of elevations
            
            ds_top = ds.isel(elev=e_top)
            ds_bot = ds.isel(elev=e_bot)
            
            Z_top, Z_bot = ds_top.Z.values[NaN_reg], ds_bot.Z.values[NaN_reg]
            QI_top, QI_bot = ds_top.QI.values[NaN_reg], ds_bot.QI.values[NaN_reg]
            
            # 2. Compute Z weighting with top and bottom values
            numerator = np.nansum([Z_top*QI_top, Z_bot*QI_bot], axis=0)
            denominator = np.nansum([QI_top, QI_bot], axis=0)
            denominator[denominator==0] = np.nan # Assign NaN values where QI = 0 in both top and bottom
            CAPPI_reg = numerator / denominator

            # 3. Handle QI = 0 by doing mean
            num = np.nansum([Z_top, Z_bot], axis=0)
            CAPPI_reg[np.isnan(denominator)] = num[np.isnan(denominator)] / 2

            # 4. Compute new QI by doing mean
            numerator = np.nansum([QI_top, QI_bot], axis=0)
            QI_reg =  numerator / 2

            CAPPI[NaN_reg] = CAPPI_reg
            QI[NaN_reg] = QI_reg
            ELEV[NaN_reg] = ds.elev.values[e_bot]
    
    # Uncomment to check if there are any NaN values in the region
    # reg_1 = ds.isel(elev=-1).H.values > CAPPI_H
    # reg_2 = ds.isel(elev=0).H.values < CAPPI_H
    # # print(np.any(np.isnan(CAPPI[reg_1*reg_2])))

    # =========================== VALUES "OUTSIDE" LOWEST BEAM ===========================
    e = 0
    # print(f"\nOuter layer: {ds.elev.values[e]}")
    ds_e = ds.isel(elev=e)

    # region further from the radar
    H_top = ds_e.H.values
    reg = (H_top > CAPPI_H) * (np.isnan(ds_e.Z.values) == 0)

    # project Z & QI and adjust QI
    # dis_top = H_top - CAPPI_H
    # w_dis_top = distance_weighting(dis_top)
    CAPPI[reg] = ds_e.Z.values[reg]
    QI[reg] = ds_e.QI.values[reg] # * w_dis_top[reg]
    ELEV[reg] = e

    # handle QI=0 values projected, it projects from the beam above with available data
    # print()
    while np.any(QI[reg] <= QI_th) and e < len(ds.elev.values)-1:
        NaN_reg = reg * (QI <= QI_th)

        e += 1
        # print(f"Outer layer: {ds.elev.values[e]}")
        ds_e = ds.isel(elev=e)

        NaN_reg_improv = NaN_reg * (ds_e.QI.values > QI)

        # H_top = ds_e.H.values
        # dis_top = H_top - CAPPI_H
        # w_dis_top = distance_weighting(dis_top)
        CAPPI[NaN_reg_improv] = ds_e.Z.values[NaN_reg_improv]
        QI[NaN_reg_improv] = ds_e.QI.values[NaN_reg_improv] # * w_dis_top[NaN_reg]
        ELEV[NaN_reg_improv] = e

    return CAPPI, QI, ELEV


def make_LUE(ds, DEM_resampled):
    # Define bad quality threshold
    QI_th = 0.1

    # Extract lowest elevation variables (reflectivity, height to ground and quality index)
    LUE = ds.isel(elev=0).Z.values
    H = ds.isel(elev=0).H.values - DEM_resampled
    QI = ds.isel(elev=0).QI.values

    # Weight quality considering height
    QI[LUE != -32] = QI[LUE != -32] * distance_weighting(H)[LUE != -32]

    # Initialize the elevation array which will store what PPI has been used
    ELEV = np.ones_like(LUE) * np.nan
    ELEV[np.isnan(LUE)==0] = ds.elev.values[0]

    # Iterate through all elevations while there is a quality index lower than threshold
    e = 1
    while (e < len(ds.elev.values)) and np.any(QI <= QI_th):
        # Assign new variables for current elevation and weight QI with height
        Z_e = ds.isel(elev=e).Z.values
        QI_e = ds.isel(elev=e).QI.values
        QI_e[LUE != -32] = QI_e[LUE != -32] * distance_weighting(H)[LUE != -32]

        # Pixels where QI <= QI_th and Z_e > Z will be redefined with current elevation on all arrays
        redef_reg = (QI <= QI_th) * (QI_e > QI_th) * (Z_e > LUE)
        ELEV[redef_reg] = ds.elev.values[e]
        H[redef_reg] = ds.isel(elev=e).H.values[redef_reg] - DEM_resampled[redef_reg]
        LUE[redef_reg] = Z_e[redef_reg]
        QI[redef_reg] = QI_e[redef_reg]

        e += 1

    # Finally, if there are still QI <= QI_th, the lowest elevation will be used
    # ELEV[QI <= QI_th] = ds.elev.values[0]
    # H[QI <= QI_th] = ds.isel(elev=0).H.values[QI <= QI_th] - DEM_resampled[QI <= QI_th]
    # LUE[QI <= QI_th] = ds.isel(elev=0).Z.values[QI <= QI_th]
    # QI[QI <= QI_th] = ds.isel(elev=0).QI.values[QI <= QI_th] * distance_weighting(H)[QI <= QI_th]

    return LUE, QI, H, ELEV