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
    
    return Z_COMP, QI_COMP, RAD_COMP, ELEV_COMP