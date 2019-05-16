import datetime_conversion
import pandas as pd
import numpy as np
import slugify
import time
import os

def dataframe_to_arrays(df, timestamp_column, object_id_column, anchor_id_column, rssi_column):
    timestamps = np.sort(df[timestamp_column].unique())
    anchor_ids = np.sort(df[anchor_id_column].unique())
    object_ids = np.sort(df[object_id_column].unique())
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
    for group_name, df_single_timestamp in df.groupby(timestamp_column):
        if time_index % 1000 == 0:
            print('Time step: {}'.format(time_index))
        for row_index in range(len(df_single_timestamp)):
            object_id = df_single_timestamp.iloc[row_index][object_id_column]
            anchor_id = df_single_timestamp.iloc[row_index][anchor_id_column]
            rssi = df_single_timestamp.iloc[row_index]['rssi']
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

def dataframe_to_arrays_by_object(df, timestamp_column, object_id_column, anchor_id_column, rssi_column):
    arrays_dict = {}
    for group_name, df_single_object in df.groupby(object_id_column):
        object_id = group_name
        print('Processing data for object {} ({} rows)...'.format(object_id, len(df_single_object)))
        arrays_dict[object_id] = dataframe_to_arrays(
            df = df_single_object,
            timestamp_column = timestamp_column,
            object_id_column = object_id_column,
            anchor_id_column = anchor_id_column,
            rssi_column = rssi_column
        )
    return arrays_dict

def csv_files_by_anchor_to_dataframe(directory, filenames, anchor_ids, timestamp_column, object_id_column, anchor_id_column, rssi_column):
    num_filenames = len(filenames)
    num_anchor_ids = len(anchor_ids)
    if num_filenames != num_anchor_ids:
        raise ValueError('Filename list has {} elements but anchor ID list has {} elements'.format(
            num_filenames,
            num_anchor_ids
        ))
    dfs = []
    print('Processing data from {} files'.format(num_filenames))
    for filename_index, filename in enumerate(filenames):
        anchor_id = anchor_ids[filename_index]
        path = os.path.join(
            directory,
            filename
        )
        df = pd.read_csv(path, parse_dates = [timestamp_column])
        start = df[timestamp_column].min()
        end = df[timestamp_column].max()
        print('{} (anchor ID {}): {} to {} ({} rows)'.format(filename, anchor_id, start, end, len(df.index)))
        df[anchor_id_column] = anchor_id
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index = True)
    df_all = df_all[[timestamp_column, object_id_column, anchor_id_column, rssi_column]]
    df_all.sort_values(timestamp_column, inplace=True)
    df_all.reset_index(inplace = True, drop = True)
    return df_all

def dataframe_to_pkl_csv_files(df, directory, filename_stem):
    pickle_path = os.path.join(
        directory, filename_stem + '.pkl')
    csv_path = os.path.join(
        directory, filename_stem + '.csv')
    print('Writing to {}'.format(pickle_path))
    df.to_pickle(pickle_path)
    print('Writing to {}'.format(csv_path))
    df.to_csv(csv_path, index = False)

def dataframe_to_pkl_csv_files_by_object(df, directory, filename_stem, object_id_column):
    for object_id, df_single_object in df.groupby(object_id_column):
        extended_filename_stem = filename_stem + '_' + slugify.slugify(object_id)
        dataframe_to_files(
            df = df_single_object,
            directory = directory,
            filename_stem = extended_filename_stem
        )

def arrays_to_npz_file(arrays, directory, filename_stem):
    npz_path = os.path.join(
        directory, filename_stem + '.npz')
    print('Writing to {}'.format(npz_path))
    np.savez_compressed(npz_path, **arrays)

def arrays_by_object_to_npz_files_by_object(arrays_by_object, directory, filename_stem):
    for object_id, arrays in arrays_by_object.items():
        extended_filename_stem = filename_stem + '_' + slugify.slugify(object_id)
        arrays_to_file(
            arrays = arrays,
            directory = directory,
            filename_stem = extended_filename_stem
        )

def npz_file_to_arrays(directory, filename):
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

def get_object_info_from_csv_file(object_ids, directory, filename, object_id_column, object_name_column, fixed_object_positions_columns):
    object_info_df = pd.DataFrame.from_dict({'object_id': object_ids})
    path = os.path.join(directory, filename)
    file_df = pd.read_csv(path)
    object_info_df = object_info_df.merge(
        right = file_df,
        how = 'left',
        left_on = 'object_id',
        right_on = object_id_column)
    object_names = object_info_df[object_name_column].values.tolist()
    fixed_object_positions = object_info_df[fixed_object_positions_columns].values
    object_info = {
        'object_names': object_names,
        'fixed_object_positions': fixed_object_positions
    }
    return object_info
