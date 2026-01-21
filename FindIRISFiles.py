import os
import datetime as dt
import numpy as np

def search_path(time_dt, directory):
    next_time = time_dt + dt.timedelta(minutes=6)

    rads = ['CDV', 'PBE', 'PDA', 'LMI']
    paths = {}
    for rad_abr in rads:
        folder_name = f"{rad_abr}RAW{time_dt.strftime("%Y%m%d")}"
        folder_dir = f"{directory}{folder_name}/"
        rad_times, rad_paths = [], []

        try:
            # Loop through all entries in the directory
            for filename in os.listdir(folder_dir):
                file_time = dt.datetime.strptime(filename[:15], f"{rad_abr}%y%m%d%H%M%S")
                if time_dt <= file_time and file_time < next_time:
                    rad_times.append(file_time)
                    rad_paths.append(f"{folder_dir}{filename}")
            
            paths[rad_abr] = {"times": rad_times, "paths": rad_paths}
        
        except Exception as e:
            print(e)
    
    return paths

def search_long_range(time, directory):
    year, month, day, hour, min = time
    time_dt = dt.datetime(year, month, day, hour, min)
    
    paths = search_path(time_dt, directory)

    VOLA_paths = []
    for rad in ['CDV', 'PBE', 'PDA', 'LMI']:
        VOLA = paths[rad]['paths'][np.argmin(paths[rad]['times'])]
        VOLA_paths.append(VOLA)
    
    return VOLA_paths

def search_short_range(time, directory):
    year, month, day, hour, min = time
    time_dt = dt.datetime(year, month, day, hour, min)

    paths = search_path(time_dt, directory)

    VOLBC_paths = []
    for rad in ['CDV', 'PBE', 'PDA', 'LMI']:
        VOLA = paths[rad]['paths'][np.argmin(paths[rad]['times'])]
        VOLC = paths[rad]['paths'][np.argmax(paths[rad]['times'])]
        VOLB = [p for p in paths[rad]['paths'] if p != VOLA and p != VOLC][0]

        VOLBC_paths.append(VOLB)
        VOLBC_paths.append(VOLC)

    return VOLBC_paths