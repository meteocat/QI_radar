import datetime as dt
import os

def load_config(config_file):
    with open(config_file, "r") as f:
        config_data = f.read()

    config_lines = config_data.split("\n")
    config = {}
    for l in range(1,len(config_lines)+1):
        line = config_lines[l-1]
        
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
                raise ValueError("VOLUME in config.txt must be one of: 'VOLA', 'VOLB', 'VOLBC'")

        elif l == 14: config["CAPPI_H"] = line.strip()
        elif l == 17: 
            config["dl"] = line.strip()
            try:
                config["CAPPI_H"] = int(config["CAPPI_H"])
                config["dl"] = int(config["dl"])
            except:
                raise ValueError("CAPPI HEIGHT and CARTESIAN RESOLUTION in config.txt must be an integer value in meters.")

        elif l == 20: config["SR_DEM_path"] = line.strip()
        elif l == 23: 
            config["LR_DEM_path"] = line.strip()
            try:
                with open(config["SR_DEM_path"], "r") as f:
                    pass
                with open(config["LR_DEM_path"], "r") as f:
                    pass
            except:
                raise ValueError("DEM file path(s) in config.txt is/are incorrect.")

        elif l == 26: 
            config["PPI_save_dir"] = line.strip()
            os.makedirs(config["PPI_save_dir"], exist_ok=True)

        elif l == 29: 
            config["TOP12_clim_path"] = line.strip()
            try:
                with open(config["TOP12_clim_path"], "r") as f:
                    pass
            except:
                raise ValueError("TOP12 climatology file path in config.txt is incorrect.")
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    config["IRIS_dir"] = f"{current_path}/data/raw/"
    if not os.path.exists(config["IRIS_dir"]) or len(os.listdir(config["IRIS_dir"])) == 0:
        raise ValueError("IRIS directory path does not exist or is empty. Please create a 'data/raw' folder in the project directory and populate it with IRIS data.")

    config["product_save_dir"] = f"{current_path}/data/processed/"
    os.makedirs(config["product_save_dir"], exist_ok=True)
        
    return config