import os
import datetime as dt
import numpy as np
import xradar as xd

def search_path(time_dt: dt.datetime, directory: str) -> dict:
    ''' Search for radar files in the specified directory within a 6-minute window.
    
    :param time_dt: datetime object representing the target time
    :param directory: Directory path where radar files are stored
    :return: Dictionary with radar abbreviations as keys and their corresponding times and paths
    '''

    # Define the next time step to use as threshold (6 minutes later)
    next_time = time_dt + dt.timedelta(minutes=6)

    # List of radar abbreviations to search for
    rads = ['CDV', 'PBE', 'PDA', 'LMI']

    # Initialize dictionary to hold paths and iterate over radar codes
    paths = {}
    for rad_abr in rads:
        # Construct folder name to search
        folder_name = f"{rad_abr}RAW{time_dt.strftime('%Y%m%d')}"
        folder_dir = f"{directory}{folder_name}/"
        rad_times, rad_paths = [], []

        try:
            # Loop through all entries in the directory
            for filename in os.listdir(folder_dir):

                # Extract file time from filename
                file_time = dt.datetime.strptime(filename[:15], f"{rad_abr}%y%m%d%H%M%S")

                # If file time is within the specified window, store its time and path
                if time_dt <= file_time and file_time < next_time:
                    rad_times.append(file_time)
                    rad_paths.append(f"{folder_dir}{filename}")
            
            # Store the found times and paths in the dictionary
            paths[rad_abr] = {"times": rad_times, "paths": rad_paths}
        
        except Exception as e:
            paths[rad_abr] = {"times": [None for i in range(3)], "paths": [None for i in range(3)]}
    
    return paths

def search_long_range(time: tuple, directory: str) -> list:
    ''' Filter and return the VOLA radar file paths for the specified time.
    
    :param time: Tuple of (year, month, day, hour, min)
    :param directory: Directory path where radar files are stored

    :return: List of VOLA radar file paths
    '''

    # Convert time tuple to datetime object
    year, month, day, hour, min = time
    time_dt = dt.datetime(year, month, day, hour, min)
    
    # Search for radar files in the specified directory
    paths = search_path(time_dt, directory)

    # Extract the VOLA files (earliest files for each radar)
    VOLA_paths = []
    for rad in ['CDV', 'PBE', 'PDA', 'LMI']:
        VOLA = None
        for p in paths[rad]['paths']:
            if p != None:
                vol = xd.io.open_iris_datatree(p, reindex_angle=False)
                scan_name = vol.attrs['scan_name']

                if "A" in scan_name: 
                    VOLA = p
                    break

        # VOLA = paths[rad]['paths'][np.argmin(paths[rad]['times'])]
        VOLA_paths.append(VOLA)
    
    return VOLA_paths

def search_short_range(time: tuple, directory: str) -> list:
    ''' Filter and return the VOLBC radar file paths for the specified time.

    :param time: Tuple of (year, month, day, hour, min)
    :param directory: Directory path where radar files are stored

    :return: List of VOLBC radar file paths
    '''

    # Convert time tuple to datetime object
    year, month, day, hour, min = time
    time_dt = dt.datetime(year, month, day, hour, min)

    # Search for radar files in the specified directory
    paths = search_path(time_dt, directory)

    # Extract the VOLBC files (latest two files for each radar)
    VOLBC_paths = []
    for rad in ['CDV', 'PBE', 'PDA', 'LMI']:
        VOLB, VOLC = None, None
        for p in paths[rad]['paths']:
            if p != None:
                vol = xd.io.open_iris_datatree(p, reindex_angle=False)
                scan_name = vol.attrs['scan_name']

                if "B" in scan_name: VOLB = p
                if "C" in scan_name: VOLC = p

        # VOLA = paths[rad]['paths'][np.argmin(paths[rad]['times'])]
        # VOLC = paths[rad]['paths'][np.argmax(paths[rad]['times'])]
        # VOLB = [p for p in paths[rad]['paths'] if p != VOLA and p != VOLC][0]

        VOLBC_paths.append(VOLB)
        VOLBC_paths.append(VOLC)

    return VOLBC_paths

# TESTING FUNCTIONS

# paths = search_path(dt.datetime(2026, 9, 21, 12, 0), "/home/nvm/nvm_local/data/rad_data/")
# for rad in paths:
#     print(f"{rad}:")
#     for time, path in zip(paths[rad]['times'], paths[rad]['paths']):
#         print(f"  Time: {time}, Path: {path}")

# paths = search_short_range((2010, 2, 4, 12, 0), "/home/nvm/nvm_local/data/rad_data/")
# print("VOLBC paths:")
# for path in paths:
#     print(f"  {path}")