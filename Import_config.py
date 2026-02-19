import datetime as dt
import os

def load_config(config_file: str) -> dict:
    '''Load configuration parameters from a text file.
    Parameters
    ----------
    config_file : str
        Path to the configuration text file.

    Returns
    -------
    dict
        Dictionary containing parsed and validated configuration values.
    '''

    # Read the configuration file into a single string. Propagate a
    # clear FileNotFoundError if the file cannot be opened.
    try:
        with open(config_file, "r") as f:
            config_data = f.read()
    except Exception:
        raise FileNotFoundError(
            "Configuration file 'config.txt' not found. Please create it based on 'config_template.txt'."
        )

    # Parse and validate expected configuration values from specific lines.
    config_lines = config_data.split("\n")
    config = {}
    for l in range(1,len(config_lines)+1):
        line = config_lines[l-1] # Adjust for 0-based index
        
        # Parse expected configuration values based on line number

        if l == 5: config["init_dt"] = line.strip()
        elif l == 8: 
            config["fin_dt"] = line.strip()

            try:
                config["init_dt"] = dt.datetime.strptime(config["init_dt"], "%Y-%m-%dT%H:%M")
                config["fin_dt"] = dt.datetime.strptime(config["fin_dt"], "%Y-%m-%dT%H:%M")
            except:
                raise ValueError("Date format in config.txt is incorrect. Use YYYY-MM-DDTHH:MM")

        elif l == 11: 
            config["VOLUME"] = line.strip()
            if config["VOLUME"] not in ['VOLA', 'VOLB', 'VOLBC']:
                raise ValueError("VOLUME SCAN USED in config.txt must be one of: 'VOLA', 'VOLB', 'VOLBC'")
        
        elif l == 14:
            config["PROD_types"] = line.strip().split(", ")
            if not all(comp in ["LUE", "CAPPI"] for comp in config["PROD_types"]):
                raise ValueError("PRODUCT TYPES in config.txt must be a comma-separated list of 'LUE' and/or 'CAPPI'")


        elif l == 17: 
            config["COMP_types"] = line.strip().split(", ")
            if not all(comp in ["MAXZ", "MAXQI", "MAXQCOND"] for comp in config["COMP_types"]):
                raise ValueError("COMPOSITE TYPES in config.txt must be a comma-separated list of 'MAXZ', 'MAXQI' and/or 'MAXQCOND'")

        elif l == 20: config["CAPPI_H"] = line.strip()
        elif l == 23: 
            config["dl"] = line.strip()
            try:
                config["CAPPI_H"] = int(config["CAPPI_H"])
                config["dl"] = int(config["dl"])
            except:
                raise ValueError("CAPPI HEIGHT and CARTESIAN RESOLUTION in config.txt must be an integer value in meters.")

        elif l == 26: 
            config["IRIS_dir"] = line.strip()
            if not os.path.exists(config["IRIS_dir"]) or len(os.listdir(config["IRIS_dir"])) == 0:
                raise ValueError("RAW DATA DIRECTORY does not exist or is empty. Please create a 'data/raw' folder in the project directory and populate it with IRIS data.")
        
        elif l == 29: 
            config["product_save_dir"] = line.strip()
            try:
                os.makedirs(config["product_save_dir"], exist_ok=True)
            except:
                raise ValueError("PROCESSED netCDF DIRECTORY in config.txt is incorrect.")

        elif l == 32: 
            config["png_save_dir"] = line.strip()
            if config["png_save_dir"] != "":
                try:
                    os.makedirs(config["png_save_dir"], exist_ok=True)
                except:
                    raise ValueError("PROCESSED PNG DIRECTORY in config.txt is incorrect.")
            
        elif l == 35: config["SR_DEM_path"] = line.strip()

        elif l == 38: 
            config["LR_DEM_path"] = line.strip()
            try:
                with open(config["SR_DEM_path"], "r") as f:
                    pass
                with open(config["LR_DEM_path"], "r") as f:
                    pass
            except:
                raise ValueError("DEM file path(s) in config.txt is/are incorrect.")

        elif l == 41: 
            config["PPI_save_dir"] = line.strip()
            os.makedirs(config["PPI_save_dir"], exist_ok=True)

        elif l == 44: 
            config["TOP12_clim_path"] = line.strip()
            try:
                with open(config["TOP12_clim_path"], "r") as f:
                    pass
            except:
                raise ValueError("TOP12 climatology file path in config.txt is incorrect.")
        
    return config