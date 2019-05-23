import smcmodel_localize.model
from smcmodel.databases.memory import DatabaseMemory
import datetime_conversion
import pandas as pd
import numpy as np
import slugify
import time
import os

def csv_file_to_dataframe(
    directory,
    filename,
    timestamp_column_name = 'timestamp'
):
    path = os.path.join(directory, filename)
    dataframe = pd.read_csv(path, parse_dates = [timestamp_column_name])
    return dataframe

def add_ids_to_dataframe(
    dataframe,
    **kwargs
):
    for column_name, column_value in kwargs.items():
        dataframe[column_name] = column_value
    return dataframe

def csv_file_to_dataframe_add_ids(
    directory,
    filename,
    timestamp_column_name = 'timestamp',
    **kwargs
):
    dataframe = csv_file_to_dataframe(directory, filename, timestamp_column_name)
    dataframe = add_ids_to_dataframe(dataframe, **kwargs)
    return dataframe

def csv_files_to_dataframe(
    directories,
    filename_parser = None,
    anchor_ids = None,
    object_ids = None,
    start_timestamp = None,
    end_timestamp = None,
    timestamp_column_name = 'timestamp',
    object_id_column_name = 'object_id',
    anchor_id_column_name = 'anchor_id',
    rssi_column_name = 'rssi'
):
    dataframes = []
    for directory in directories:
        filenames = os.listdir(directory)
        for filename in filenames:
            if filename_parser is not None:
                ids = filename_parser(filename)
                if ids is not None:
                    dataframe = csv_file_to_dataframe(
                        directory,
                        filename,
                        timestamp_column_name)
                    dataframe = add_ids_to_dataframe(
                        dataframe,
                        **ids
                    )
                    dataframe = filter_dataframe(
                        dataframe,
                        anchor_ids,
                        object_ids,
                        start_timestamp,
                        end_timestamp,
                        timestamp_column_name,
                        object_id_column_name,
                        anchor_id_column_name
                    )
                    dataframes.append(dataframe)
            else:
                    dataframe = csv_file_to_dataframe(
                        directory,
                        filename,
                        timestamp_column_name)
                    dataframe = filter_dataframe(
                        dataframe,
                        anchor_ids,
                        object_ids,
                        start_timestamp,
                        end_timestamp,
                        timestamp_column_name,
                        object_id_column_name,
                        anchor_id_column_name
                    )
                    dataframes.append(dataframe)
    dataframe_all = pd.concat(dataframes, ignore_index = True)
    dataframe_all = dataframe_all[[timestamp_column_name, object_id_column_name, anchor_id_column_name, rssi_column_name]]
    dataframe_all.sort_values(timestamp_column_name, inplace=True)
    dataframe_all.reset_index(inplace = True, drop = True)
    return dataframe_all


def csv_files_by_anchor_to_dataframe(
    directory,
    filenames,
    anchor_ids,
    timestamp_column_name = 'timestamp',
    object_id_column_name = 'object_id',
    anchor_id_column_name = 'anchor_id',
    rssi_column_name = 'rssi'
):
    num_filenames = len(filenames)
    num_anchor_ids = len(anchor_ids)
    if num_filenames != num_anchor_ids:
        raise ValueError('Filename list has {} elements but anchor ID list has {} elements'.format(
            num_filenames,
            num_anchor_ids
        ))
    dataframes = []
    print('Processing data from {} files'.format(num_filenames))
    for filename_index, filename in enumerate(filenames):
        anchor_id = anchor_ids[filename_index]
        dataframe = csv_file_to_dataframe_add_ids(
            directory = directory,
            filename = filename,
            timestamp_column_name = timestamp_column_name,
            **{anchor_id_column_name: anchor_id})
        start = dataframe[timestamp_column_name].min()
        end = dataframe[timestamp_column_name].max()
        print('{} (anchor ID {}): {} to {} ({} rows)'.format(filename, anchor_id, start, end, len(dataframe.index)))
        dataframes.append(dataframe)
    dataframe_all = pd.concat(dataframes, ignore_index = True)
    dataframe_all = dataframe_all[[timestamp_column_name, object_id_column_name, anchor_id_column_name, rssi_column_name]]
    dataframe_all.sort_values(timestamp_column_name, inplace=True)
    dataframe_all.reset_index(inplace = True, drop = True)
    return dataframe_all

def filter_dataframe(
    dataframe,
    anchor_ids = None,
    object_ids = None,
    start_timestamp = None,
    end_timestamp = None,
    timestamp_column_name = 'timestamp',
    object_id_column_name = 'object_id',
    anchor_id_column_name = 'anchor_id'
):
    if anchor_ids is not None:
        anchor_id_boolean = dataframe[anchor_id_column_name].isin(anchor_ids)
    else:
        anchor_id_boolean = True
    if object_ids is not None:
        object_id_boolean = dataframe[object_id_column_name].isin(object_ids)
    else:
        object_id_boolean = True
    if start_timestamp is not None:
        start_timestamp_boolean = (dataframe[timestamp_column_name] >= start_timestamp)
    else:
        start_timestamp_boolean = True
    if end_timestamp is not None:
        end_timestamp_boolean = (dataframe[timestamp_column_name] <= end_timestamp)
    else:
        end_timestamp_boolean = True
    combined_boolean = anchor_id_boolean & object_id_boolean & start_timestamp_boolean & end_timestamp_boolean
    if combined_boolean != True:
        dataframe_filtered = dataframe[combined_boolean]
    else:
        dataframe_filtered = dataframe
    return dataframe_filtered

def dataframe_to_arrays(
    dataframe,
    timestamp_column_name = 'timestamp',
    object_id_column_name = 'object_id',
    anchor_id_column_name = 'anchor_id',
    rssi_column_name = 'rssi'
):
    timestamps = np.sort(dataframe[timestamp_column_name].unique())
    anchor_ids = np.sort(dataframe[anchor_id_column_name].unique())
    object_ids = np.sort(dataframe[object_id_column_name].unique())
    anchor_id_list = anchor_ids.tolist()
    object_id_list = object_ids.tolist()
    anchor_indices = {anchor_id: anchor_index for anchor_index, anchor_id in enumerate(anchor_id_list)}
    object_indices = {object_id: object_index for object_index, object_id in enumerate(object_id_list)}
    num_timestamps = len(timestamps)
    num_anchors = len(anchor_ids)
    num_objects = len(object_ids)
    rssis = np.full((num_timestamps, 1, num_anchors, num_objects), np.nan, dtype=np.float32)
    time_index = 0
    print('Processing {} time steps...'.format(num_timestamps))
    start_time = time.time()
    for group_name, dataframe_single_timestamp in dataframe.groupby(timestamp_column_name):
        if time_index % 1000 == 0:
            print('Time step: {}'.format(time_index))
        for row_index in range(len(dataframe_single_timestamp)):
            object_id = dataframe_single_timestamp.iloc[row_index][object_id_column_name]
            anchor_id = dataframe_single_timestamp.iloc[row_index][anchor_id_column_name]
            rssi = dataframe_single_timestamp.iloc[row_index]['rssi']
            if object_id in object_ids and anchor_id in anchor_ids:
                anchor_index = anchor_indices[anchor_id]
                object_index = object_indices[object_id]
                rssis[time_index, 0, anchor_index, object_index] = rssi
        time_index += 1
    total_time = time.time() - start_time
    print('Processed {} time steps in {:.4} seconds ({:.4} ms per time step)'.format(
        num_timestamps,
        total_time,
        total_time*1000/num_timestamps
    ))
    arrays = {
        'num_timestamps': num_timestamps,
        'num_anchors': num_anchors,
        'num_objects': num_objects,
        'timestamps': timestamps,
        'anchor_ids': anchor_ids,
        'object_ids': object_ids,
        'rssis': rssis
    }
    return arrays

def dataframe_to_arrays_by_object(
    dataframe,
    timestamp_column_name = 'timestamp',
    object_id_column_name = 'object_id',
    anchor_id_column_name = 'anchor_id',
    rssi_column_name = 'rssi'
):
    arrays_dict = {}
    for group_name, dataframe_single_object in dataframe.groupby(object_id_column_name):
        object_id = group_name
        print('Processing data for object {} ({} rows)...'.format(object_id, len(dataframe_single_object)))
        arrays_dict[object_id] = dataframe_to_arrays(
            dataframe = dataframe_single_object,
            timestamp_column_name = timestamp_column_name,
            object_id_column_name = object_id_column_name,
            anchor_id_column_name = anchor_id_column_name,
            rssi_column_name = rssi_column_name
        )
    return arrays_dict

def arrays_to_observation_database(arrays):
    observation_structure = smcmodel_localize.model.observation_structure_generator(arrays['num_anchors'], arrays['num_objects'])
    observation_time_series_data = {'rssis': arrays['rssis']}
    observation_database = DatabaseMemory(
        structure = observation_structure,
        num_samples = 1,
        timestamps = arrays['timestamps'],
        time_series_data = observation_time_series_data)
    return observation_database

def dataframe_to_pkl_csv_files(
    dataframe,
    directory,
    filename_stem
):
    pickle_path = os.path.join(
        directory, filename_stem + '.pkl')
    csv_path = os.path.join(
        directory, filename_stem + '.csv')
    print('Writing to {}'.format(pickle_path))
    dataframe.to_pickle(pickle_path)
    print('Writing to {}'.format(csv_path))
    dataframe.to_csv(csv_path, index = False)

def dataframe_to_pkl_csv_files_by_object(
    dataframe,
    directory,
    filename_stem,
    object_id_column_name = 'object_id'
):
    for object_id, dataframe_single_object in dataframe.groupby(object_id_column_name):
        extended_filename_stem = filename_stem + '_' + slugify.slugify(object_id)
        dataframe_to_pkl_csv_files(
            dataframe = dataframe_single_object,
            directory = directory,
            filename_stem = extended_filename_stem
        )

def arrays_to_npz_file(
    arrays,
    directory,
    filename_stem
):
    npz_path = os.path.join(
        directory, filename_stem + '.npz')
    print('Writing to {}'.format(npz_path))
    np.savez_compressed(npz_path, **arrays)

def arrays_by_object_to_npz_files_by_object(
    arrays_by_object,
    directory,
    filename_stem
):
    for object_id, arrays in arrays_by_object.items():
        extended_filename_stem = filename_stem + '_' + slugify.slugify(object_id)
        arrays_to_npz_file(
            arrays = arrays,
            directory = directory,
            filename_stem = extended_filename_stem
        )

def npz_file_to_arrays(
    directory,
    filename
):
    path = os.path.join(directory, filename)
    npz_data = np.load(path)
    data = {
        'num_anchors': npz_data['num_anchors'].item(),
        'num_objects': npz_data['num_objects'].item(),
        'num_timestamps': npz_data['num_timestamps'].item(),
        'anchor_ids': npz_data['anchor_ids'].tolist(),
        'object_ids': npz_data['object_ids'].tolist(),
        'timestamps': datetime_conversion.to_posix_timestamps(npz_data['timestamps']),
        'rssis': np.asarray(npz_data['rssis'])
    }
    return data

def get_object_info_from_csv_file(
    object_ids,
    directory,
    filename,
    fixed_object_positions_column_names,
    object_name_column_name = None,
    object_id_column_name = 'object_id'
):
    object_info_dataframe = pd.DataFrame.from_dict({'object_id': object_ids})
    path = os.path.join(directory, filename)
    file_dataframe = pd.read_csv(path)
    object_info_dataframe = object_info_dataframe.merge(
        right = file_dataframe,
        how = 'left',
        left_on = 'object_id',
        right_on = object_id_column_name)
    fixed_object_positions = object_info_dataframe[fixed_object_positions_column_names].values
    object_info = {
        'fixed_object_positions': fixed_object_positions
    }
    if object_name_column_name is not None:
        object_names = object_info_dataframe[object_name_column_name].values.tolist()
        object_info['object_names'] = object_names
    return object_info

def get_anchor_info_from_csv_file(
    anchor_ids,
    directory,
    filename,
    anchor_positions_column_names,
    anchor_name_column_name = None,
    anchor_id_column_name = 'anchor_id'
):
    anchor_info_dataframe = pd.DataFrame.from_dict({'anchor_id': anchor_ids})
    path = os.path.join(directory, filename)
    file_dataframe = pd.read_csv(path)
    anchor_info_dataframe = anchor_info_dataframe.merge(
        right = file_dataframe,
        how = 'left',
        left_on = 'anchor_id',
        right_on = anchor_id_column_name)
    anchor_positions = anchor_info_dataframe[anchor_positions_column_names].values
    anchor_info = {
        'anchor_positions': anchor_positions
    }
    if anchor_name_column_name is not None:
        anchor_names = anchor_info_dataframe[anchor_name_column_name].values.tolist()
        anchor_info['anchor_names'] = anchor_names
    return anchor_info
