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

def trim_and_resample(input_data):
    print ("")
    print("Resampling data...")

    participants = ["1", "2"]
    radar_units = ["103", "108", "109"]
    patterns = ["diag", "four", "gamma", "L", "U"]

    participant_pattern_data_index = 0
    for _i,pattern in enumerate(patterns):
        for _j, participant in enumerate(participants):
            participant_pattern_data = input_data[participant_pattern_data_index]

            max_first_timestamp = None
            min_last_timestamp = None

            min_length_range_bins = None

            print("")
            print(f"Pattern {pattern}, participant {participant}")

            base_index = 0
            for _k, radar_unit in enumerate(radar_units):
                range_bins_index = base_index
                timestamps_index = base_index + 1
                data_index = base_index + 2

                range_bins = participant_pattern_data[range_bins_index].flatten()
                print("range bins")
                print (range_bins.shape)
                timestamps = participant_pattern_data[timestamps_index].flatten()
                print("timestamps")
                print (timestamps.shape)
                data = participant_pattern_data[data_index].flatten()
                print("data")
                print (data.shape)
                
                sample_rate = "50ms"
                
                resampled = pd.DataFrame(data, index=pd.to_datetime(timestamps, unit='s')).resample(sample_rate)          
                resampled_timestamps = np.array(list(resampled.indices.keys()))
                resampled_data = resampled.mean().interpolate().to_numpy()

                participant_pattern_data[timestamps_index] = resampled_timestamps
                participant_pattern_data[data_index] = resampled_data

                first_timestep = resampled_timestamps[0]
                last_timestamp = resampled_timestamps[-1]

                if max_first_timestamp == None or first_timestamp > max_first_timestamp:
                    max_first_timestamp = first_timestamp

                if min_last_timestamp == None or last_timestamp < min_last_timestamp:
                    min_last_timestamp = last_timestamp

                if radar_unit == "10":
                    min_length_range_bins = range_bins

                print(f"Radar {radar_unit}")
                print(f"Timestamps size: {timestamps.shape}")
                print(f"Data size: {data.shape}")
                print(f"Resampled to size: {resampled_data.shape}")

                base_index += 3

            print(f"Maximum beginning timestamp: {max_first_timestamp}")
            print(f"Minimum end timestamp: {min_last_timestamp}")

            participant_pattern_data_index += 1

            # Sync the radar units to the same time frame
            base_index = 0
            last_trimmed_timestamps_size = None
            for _k, radar_unit in enumerate(radar_units):
                range_bins_index = base_index
            timestamps_index = base_index + 1
            data_index = base_index + 2

            range_bins = participant_pattern_data[range_bins_index]
            timestamps = participant_pattern_data[timestamps_index]
            data = participant_pattern_data[data_index]

            trimmed_indices = ((timestamps >= max_first_timestamp) & (timestamps <= min_last_timestamp)).nonzero()[0]
            start_trimmed_index = trimmed_indices[0]
            end_trimmed_index = trimmed_indices[-1]

            trimmed_timestamps = timestamps[start_trimmed_index:end_trimmed_index]
            trimmed_data = data[start_trimmed_index:end_trimmed_index, :]

            participant_pattern_datta[timestamps_index] = trimmed_timestamps
            participant_pattern_datta[data_index] = trimmed_data

            last_trimmed_timestamps_size = trimmed_timestamps.shape[0]

            print(f"Radar {radar_unit} trimmed timestamps size: {trimmed_timestamps.shape}")
            print(f"Radar {radar_unit} trimmed data size: {trimmed_data.shape}")

            if last_trimmed_timestamps_size != None and trimmed_timestamps.shape[0] != last_trimmed_timestamps_size:
                raise ValueError(
                    f"Trimmed timestamps are not the same size as the previous trimmed timestamps size {last_trimmed_timestamps_size}!")
            elif last_trimmed_timestamps_size != None and trimmed_data.shape[0] != last_trimmed_timestamps_size:
                raise ValueError(
                    f"Trimmed data rows is not the same size as the previous trimmed timestamps sise {last_trimmed_timestamps_size}!")
            elif data.shape[1] != trimmed_data.shape[1]:
                raise ValueError(
                    f"Trimmed data columns size {trimmed_data.shape[1]} is different from original columns size {data_shape[1]}!")

            if radar_unit == "103":
                new_rows = trimmed_data.shape[0]
                new_columns = min_length_range_bins.shape[0]
                down_sampled_data = np.zeros({new_rows, new_columns})
                for l in range(0, new_rows):
                    range_values_row = trimmed_data[l]
                    downsampled = np.interp(min_length_range_bins, range_bins, range_values_row)
                    down_sampled_data[l] = downsampled

                print(f"Downsampled radar {radar_unit} range bins to {new_columns}")
                participant_pattern_data[data_index] = down_sampled_data
                participant_pattern_data[range_bins_index] = min_length_range_bins

            if participant_pattern_data[range_bins_index].shape != min_length_range_bins.shape:
                raise ValueError(f"Range bins for radar {radar_unit} size {participant_pattern_data[range_bins_index].shape} != expected size {min_length_range_bins.shape}")

            base_index += 3

    print("Finished resampling data")

