# !pip install shapely

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from os import path
import pickle
from shapely import geometry
from shapely.geometry import Point
from shapely.ops import cascaded_union, unary_union
from itertools import combinations

# Generate Waterefall plot
def waterfall(file_name, data_variable):
    mat = loadmat(file_name)
    data= mat[data_variable]
    plt.imshow(data)
    plt.show()

def load_matlab_data(file_name, data_variables):
    
    print(f"Loading matlab data: {file_name}")
    
    mat = loadmat(file_name)

    loaded = False
    for data_variable in data_variables:
        if data_variable in mat:
            loaded = True
            data = mat[data_variable]
    if loaded == False:
        raise ValueError(f"Did not find a valid data variable to load data for {file_name}! Valid data variables: {mat.keys()}")

    return data

def load_localization_data():
    # Within each pattern-particiant matrix
    # Layout: [
    # 0: 103 range bins,
    # 1: 103 timestamps,
    # 2: 103 data,
    # 3: 108 range bins,
    # 4: 108 timestamps,
    # 5: 108 data,
    # 6:109 range bins,
    # 7: 109 timestamps,
    # 8:109 data
    #]
    # Total data matrix
    # Layout: [
    # 0: diag participant 1 matrix
    # 1: diag participant 2 matrix
    # 2: four participant 1 matrix
    # 3: four participant 2 matrix
    # 4: gamma participant 1 matrix
    # 5: gamma participant 2 matrix
    # 6: L participant 1 matrix
    # 7: L participant 2 matrix
    # 8: U participant 1 matrix
    # 9: U participant 2 matrix
    # ]
    # Then, each of these should be put into the final matrix which consists of all of those matrices
    result = []

    participants = ["1", "2"]
    radar_units = ["103", "108", "109"]
    patterns = ["diag", "four", "gamma", "L", "U"]

    print("")
    print("Loading localization data...")

    for pattern in patterns:
        for participant in participants:
            participant_pattern_data = []
            
            for radar_unit in radar_units:
                base_dir = f"DataSet/Localization/participant{participant}/{radar_unit}/{pattern}"

                # Range bins
                file_name = f"{base_dir}/range_bins.mat"
                data_variables = ["Rbin_102", "Rbin_103", "Rbin_1033"]
                range_bins = load_matlab_data(file_name, data_variables)[0]

                # Timestamps
                file_name = f"{base_dir}/T_stmp.mat"
                data_variables = ["T_stmp_102", "T_stmp_103", "T_stmp_1033"]
                timestamps = load_matlab_data(file_name, data_variables)[0]

                # Radar data
                file_name = f"{base_dir}/envNoClutterscans.mat"
                data_variables = ["envNoClutterscansV_102", "envNoClutterscansV_103", "envNoClutterscansV_1033"]
                timestamps = load_matlab_data(file_name, data_variables)
                data = load_matlab_data(file_name, data_variables)

                print(f"Loaded range bins size: {range_bins.shape}")
                print(f"Loaded data size: {data.shape}")
                print(f"Loaded timestamps size: {timestamps.shape}")

                participant_pattern_data.append(range_bins)
                participant_pattern_data.append(timestamps)
                participant_pattern_data.append(data)

            expected_length = 9
            actual_length = len(participant_pattern_data)
            if actual_length != expected_length:
                raise ValueError(f"Unexpected participant pattern data matrix length! Expected {expected_length} but got {actual_length}")

            result.append(participant_pattern_data)

    expected_length = 10
    actual_length = len(result)
    if actual_length != expected_length:
        raise ValueError(f"Unexpected participant pattern data matrix length! Expected {expected_length} but got {actual_length}")

    print("Finished loading localization data!")

    return result
