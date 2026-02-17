import numpy as np

def composite(Z_ind_rad, QI_ind_rad, ELEV_ind_rad, comp_type="MAXZ"):
    ''' Create a composite from multiple individual radar datasets by selecting the value with the highest reflectivity or quality index for each pixel.
    
     :param Z_ind_rad: 3D array of reflectivity values for each radar, elevation and pixel
     :param QI_ind_rad: 3D array of quality index values for each radar, elevation and pixel
     :param ELEV_ind_rad: 3D array of elevation values for each radar, elevation and pixel
     :param comp_type: String indicating the composition method to use ("MAXZ" or "MAXQI")

     :return: Z_COMP, QI_COMP, RAD_COMP and ELEV_COMP arrays representing the composite reflectivity, quality index, radar source and elevation used for each pixel in the composite'''

    # Initialize composition arrays
    Z_COMP = np.ones_like(Z_ind_rad[0, ...]) * np.nan
    QI_COMP = np.ones_like(Z_ind_rad[0, ...]) * np.nan
    RAD_COMP = np.ones_like(Z_ind_rad[0, ...]) * np.nan
    ELEV_COMP = np.ones_like(Z_ind_rad[0, ...]) * np.nan
    
    # Compute composition for the method selected
    
    if comp_type == "MAXZ":
        # Iterate through each radar
        for nrad in range(len(Z_ind_rad[:,0,0])):
            # Compute region where Z is max. and where Q is max.
            reg_radZmax = Z_ind_rad[nrad, ...] > np.nan_to_num(Z_COMP, nan=-np.inf)
            reg_radQImax = QI_ind_rad[nrad, ...] > np.nan_to_num(QI_COMP, nan=-np.inf)

            # Assign radar data where Z is max.
            Z_COMP[reg_radZmax] = Z_ind_rad[nrad, ...][reg_radZmax]
            QI_COMP[reg_radZmax] = QI_ind_rad[nrad, ...][reg_radZmax]
            ELEV_COMP[reg_radZmax] = ELEV_ind_rad[nrad, ...][reg_radZmax]
            RAD_COMP[reg_radZmax] = nrad

            # Handle region where there is no detection, in which case,
            # data with the highest quality index is selected
            reg_NoZ = (Z_COMP == -32)
            QI_COMP[reg_NoZ*reg_radQImax] = QI_ind_rad[nrad, ...][reg_NoZ*reg_radQImax]
            ELEV_COMP[reg_NoZ*reg_radQImax] = ELEV_ind_rad[nrad, ...][reg_NoZ*reg_radQImax]
            RAD_COMP[reg_NoZ*reg_radQImax] = nrad

    elif comp_type == "MAXQI":
        # Iterate through each radar
        for nrad in range(len(Z_ind_rad[:,0,0])):
            # Compute region where Q is max.
            reg_radQImax = QI_ind_rad[nrad, ...] > np.nan_to_num(QI_COMP, nan=-np.inf)

            # Assign radar data where Q is max.
            Z_COMP[reg_radQImax] = Z_ind_rad[nrad, ...][reg_radQImax]
            QI_COMP[reg_radQImax] = QI_ind_rad[nrad, ...][reg_radQImax]
            ELEV_COMP[reg_radQImax] = ELEV_ind_rad[nrad, ...][reg_radQImax]
            RAD_COMP[reg_radQImax] = nrad
    
    elif comp_type == "MAXQCOND":
        th_DET = 0.6 # Quality index threshold for accepting a detection as valid
        th_UNDET = 0.7 # Quality index threshold for accepting an undetection as valid

        # ==================================== ONE-RADAR COVERAGE REGIONS ====================================

        # Region where only one radar has valid data (either a DETection or an UNDETection)
        reg_oneRadCov = np.sum(~np.isnan(Z_ind_rad), axis=0) == 1

        # Which radar has the valid data for each pixel in this region
        which_oneRadCov = np.where(reg_oneRadCov, np.argmax(~np.isnan(Z_ind_rad), axis=0), 0)

        # Assign the single radar's data to the composite in these regions
        Z_COMP[reg_oneRadCov] = np.take_along_axis(Z_ind_rad, which_oneRadCov[np.newaxis, :, :], axis=0)[0].astype(float)[reg_oneRadCov]
        QI_COMP[reg_oneRadCov] = np.take_along_axis(QI_ind_rad, which_oneRadCov[np.newaxis, :, :], axis=0)[0].astype(float)[reg_oneRadCov]
        ELEV_COMP[reg_oneRadCov] = np.take_along_axis(ELEV_ind_rad, which_oneRadCov[np.newaxis, :, :], axis=0)[0].astype(float)[reg_oneRadCov]
        RAD_COMP[reg_oneRadCov] = which_oneRadCov[reg_oneRadCov]

        # ==================================== TWO-RADAR COVERAGE REGIONS ====================================

        # Region where only one radar has a DETection (i.e. Z != -32)
        reg_singleDet = np.sum(np.nan_to_num(Z_ind_rad, nan=-32) != -32, axis=0) == 1

        # Which radar has the single DETection for each pixel
        which_rad_singleDet = np.where(reg_singleDet, np.argmax(np.nan_to_num(Z_ind_rad, nan=-32) != -32, axis=0), 0)

        # Region where only one radar has an UNDETection (i.e. Z == -32)
        reg_singleUndet = np.sum(Z_ind_rad == -32, axis=0) == 1

        # Which radar has the single UNDETection for each pixel
        which_rad_singleUndet = np.where(reg_singleUndet, np.argmax(Z_ind_rad == -32, axis=0), 0)

        # Quality of the single UNDETection radar
        QI_radUndet = np.take_along_axis(QI_ind_rad, which_rad_singleUndet[np.newaxis, :, :], axis=0)[0].astype(float)

        # Region where the single UNDETection radar has a high quality index (above the threshold)
        reg_radUndet_aboveThQI = QI_radUndet > th_UNDET

        # Region where there is a single DETection and a single UNDETection
        reg_1Det1Undet = reg_singleDet * reg_singleUndet

        # Region where there is a single DETection and a single UNDETection, and the single UNDETection radar has a high quality index
        reg_1Det1Undet_UNDET = reg_radUndet_aboveThQI * reg_1Det1Undet

        # Assign the single UNDETection radar's data to the composite in these regions
        Z_COMP[reg_1Det1Undet_UNDET] = np.take_along_axis(Z_ind_rad, which_rad_singleUndet[np.newaxis, :, :], axis=0)[0].astype(float)[reg_1Det1Undet_UNDET]
        QI_COMP[reg_1Det1Undet_UNDET] = np.take_along_axis(QI_ind_rad, which_rad_singleUndet[np.newaxis, :, :], axis=0)[0].astype(float)[reg_1Det1Undet_UNDET]
        ELEV_COMP[reg_1Det1Undet_UNDET] = np.take_along_axis(ELEV_ind_rad, which_rad_singleUndet[np.newaxis, :, :], axis=0)[0].astype(float)[reg_1Det1Undet_UNDET]
        RAD_COMP[reg_1Det1Undet_UNDET] = which_rad_singleUndet[reg_1Det1Undet_UNDET]

        # Region where there is a single DETection and a single UNDETection, and the single UNDETection radar has a low quality index
        reg_1Det1Undet_DET = (~reg_radUndet_aboveThQI) * reg_1Det1Undet

        # Assign the single UNDETection radar's data to the composite in these regions
        Z_COMP[reg_1Det1Undet_DET] = np.take_along_axis(Z_ind_rad, which_rad_singleDet[np.newaxis, :, :], axis=0)[0].astype(float)[reg_1Det1Undet_DET]
        QI_COMP[reg_1Det1Undet_DET] = np.take_along_axis(QI_ind_rad, which_rad_singleDet[np.newaxis, :, :], axis=0)[0].astype(float)[reg_1Det1Undet_DET]
        ELEV_COMP[reg_1Det1Undet_DET] = np.take_along_axis(ELEV_ind_rad, which_rad_singleDet[np.newaxis, :, :], axis=0)[0].astype(float)[reg_1Det1Undet_DET]
        RAD_COMP[reg_1Det1Undet_DET] = which_rad_singleDet[reg_1Det1Undet_DET]

        # ======================================== >1 RAD. DETECTS ========================================

        # Region where there are more than 1 DETections and 1 UNDETections
        reg_MultDet = (np.sum(np.nan_to_num(Z_ind_rad, nan=-32) != -32, axis=0) > 1)

        for nrad in range(len(Z_ind_rad[:,0,0])):
            # Compute region where Q is max.
            reg_radQImax = (QI_ind_rad[nrad, ...] > np.nan_to_num(QI_COMP, nan=-np.inf)) * (Z_ind_rad[nrad, ...] != -32) * reg_MultDet

            # Assign radar data where Q is max.
            Z_COMP[reg_radQImax] = Z_ind_rad[nrad, ...][reg_radQImax]
            QI_COMP[reg_radQImax] = QI_ind_rad[nrad, ...][reg_radQImax]
            ELEV_COMP[reg_radQImax] = ELEV_ind_rad[nrad, ...][reg_radQImax]
            RAD_COMP[reg_radQImax] = nrad

        # =============================== >1 RAD. UNDETECTS & 1 RAD. DETECTS ===============================

        # Region where there are more than 1 UNDETections (i.e. Z == -32) and ONE DETection
        reg_OneDet_MultiUndet = (np.sum(Z_ind_rad == -32, axis=0) > 1) * reg_singleDet

        # Which radar has the single DETection for each pixel in this region
        which_OneDet = np.where(reg_OneDet_MultiUndet, np.argmax(np.nan_to_num(Z_ind_rad, nan=-32) != -32, axis=0), 0)

        # Quality of the single DETection radar
        QI_OneDet = np.take_along_axis(QI_ind_rad, which_OneDet[np.newaxis, :, :], axis=0)[0].astype(float)

        # Region where there is one DETection and more than one UNDETections, and the single DETection radar has a high quality index (above the threshold)
        reg_OneDet_aboveThQI = (QI_OneDet >= th_DET) * reg_OneDet_MultiUndet

        # Assign the single DETection radar's data to the composite in these regions
        Z_COMP[reg_OneDet_aboveThQI] = np.take_along_axis(Z_ind_rad, which_OneDet[np.newaxis, :, :], axis=0)[0].astype(float)[reg_OneDet_aboveThQI]
        QI_COMP[reg_OneDet_aboveThQI] = np.take_along_axis(QI_ind_rad, which_OneDet[np.newaxis, :, :], axis=0)[0].astype(float)[reg_OneDet_aboveThQI]
        ELEV_COMP[reg_OneDet_aboveThQI] = np.take_along_axis(ELEV_ind_rad, which_OneDet[np.newaxis, :, :], axis=0)[0].astype(float)[reg_OneDet_aboveThQI]
        RAD_COMP[reg_OneDet_aboveThQI] = which_OneDet[reg_OneDet_aboveThQI]

        # Region where there is one DETection and more than one UNDETections, and the single DETection radar has a low quality index (below the threshold)
        reg_OneDet_belowThQI = (QI_OneDet < th_DET) * reg_OneDet_MultiUndet

        for nrad in range(len(Z_ind_rad[:,0,0])):
            # Compute region where Q is max.
            reg_radQImax = (QI_ind_rad[nrad, ...] > np.nan_to_num(QI_COMP, nan=-np.inf)) * (Z_ind_rad[nrad, ...] == -32) * reg_OneDet_belowThQI

            # Assign radar data where Q is max.
            Z_COMP[reg_radQImax] = Z_ind_rad[nrad, ...][reg_radQImax]
            QI_COMP[reg_radQImax] = QI_ind_rad[nrad, ...][reg_radQImax]
            ELEV_COMP[reg_radQImax] = ELEV_ind_rad[nrad, ...][reg_radQImax]
            RAD_COMP[reg_radQImax] = nrad
        
        # =============================== >1 RAD. UNDETECTS & NO DETECTS ===============================

        # Region where there are more than 1 UNDETections (i.e. Z == -32) and NO DETections
        reg_NoDet_MultiUndet = (np.sum(Z_ind_rad == -32, axis=0) > 1) * (np.sum(np.nan_to_num(Z_ind_rad, nan=-32) != -32, axis=0) == 0)

        for nrad in range(len(Z_ind_rad[:,0,0])):
            # Compute region where Q is max.
            reg_radQImax = (QI_ind_rad[nrad, ...] > np.nan_to_num(QI_COMP, nan=-np.inf)) * reg_NoDet_MultiUndet

            # Assign radar data where Q is max.
            Z_COMP[reg_radQImax] = Z_ind_rad[nrad, ...][reg_radQImax]
            QI_COMP[reg_radQImax] = QI_ind_rad[nrad, ...][reg_radQImax]
            ELEV_COMP[reg_radQImax] = ELEV_ind_rad[nrad, ...][reg_radQImax]
            RAD_COMP[reg_radQImax] = nrad
    
    return Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP