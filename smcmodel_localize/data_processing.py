import smcmodel_localize.model
from smcmodel.databases.memory import DatabaseMemory
import datetime_conversion
import pandas as pd
import numpy as np
import slugify
import time
import os
import itertools

DEFAULT_TIMESTAMP_COLUMN_NAME = 'timestamp'
DEFAULT_ANCHOR_ID_COLUMN_NAME = 'anchor_id'
DEFAULT_ANCHOR_DATA_COLUMN_NAME_PREFIX = 'anchor_'
DEFAULT_OBJECT_ID_COLUMN_NAME = 'object_id'
DEFAULT_MEASUREMENT_VALUE_COLUMN_NAME = 'value'

def wide_csv_file_to_long_csv_file(
    directory,
    filename,
    id_column_names,
    data_column_names,
    new_id_column_name,
    new_data_column_name,
    new_id_values = None,
    output_directory = '.',
    output_filename_prefix = 'long_'
):
    path = os.path.join(directory, filename)
    dataframe = pd.read_csv(path, dtype=str)
    if new_id_values is not None:
        if len(new_id_values) != len(data_column_names):
            raise ValueError('Variable value list must be same length as value column name list')
        mapping_dict = {data_column_names[i]: new_id_values[i] for i in range(len(new_id_values))}
        dataframe.rename(
            mapper = mapping_dict,
            axis = 'columns',
            inplace= True
        )
        data_column_names = new_id_values
    dataframe_long = pd.melt(
        dataframe,
        id_vars = id_column_names,
        value_vars = data_column_names,
        var_name = new_id_column_name,
        value_name = new_data_column_name
    )
    dataframe_long.dropna(
        axis = 0,
        subset = [new_data_column_name],
        inplace = True
    )
    output_filename = output_filename_prefix + filename
    output_path = os.path.join(output_directory, output_filename)
    dataframe_long.to_csv(output_path, index = False)


def csv_file_to_dataframe(
    directory,
    filename,
    timestamp_column_name = DEFAULT_TIMESTAMP_COLUMN_NAME,
    anchor_id_column_name = DEFAULT_ANCHOR_ID_COLUMN_NAME,
    object_id_column_name = DEFAULT_OBJECT_ID_COLUMN_NAME,
    measurement_value_column_name = DEFAULT_MEASUREMENT_VALUE_COLUMN_NAME
):
    path = os.path.join(directory, filename)
    dataframe = pd.read_csv(path, parse_dates = [timestamp_column_name])
    dataframe.rename(
        mapper = {
            timestamp_column_name: DEFAULT_TIMESTAMP_COLUMN_NAME,
            anchor_id_column_name: DEFAULT_ANCHOR_ID_COLUMN_NAME,
            object_id_column_name: DEFAULT_OBJECT_ID_COLUMN_NAME,
            measurement_value_column_name: DEFAULT_MEASUREMENT_VALUE_COLUMN_NAME
        },
        axis = 'columns',
        inplace = True
    )
    return dataframe

def wide_csv_file_to_dataframe(
    directory,
    filename,
    timestamp_column_name,
    anchor_data_column_names,
    object_id_column_name,
    min_data_value = None,
    max_data_value = None,
):
    num_anchor_data_columns = len(anchor_data_column_names)
    new_anchor_data_column_names = ['{}{:02}'.format(DEFAULT_ANCHOR_DATA_COLUMN_NAME_PREFIX, i) for i in range(num_anchor_data_columns)]
    column_name_mapper = {
        timestamp_column_name: DEFAULT_TIMESTAMP_COLUMN_NAME,
        object_id_column_name: DEFAULT_OBJECT_ID_COLUMN_NAME,
    }
    anchor_data_column_name_mapper = {anchor_data_column_names[i]: new_anchor_data_column_names[i] for i in range(num_anchor_data_columns)}
    column_name_mapper.update(anchor_data_column_name_mapper)
    path = os.path.join(directory, filename)
    dataframe = pd.read_csv(path, parse_dates = [timestamp_column_name])
    dataframe.rename(
        mapper = column_name_mapper,
        axis = 'columns',
        inplace = True
    )
    dataframe = dataframe[[DEFAULT_TIMESTAMP_COLUMN_NAME,DEFAULT_OBJECT_ID_COLUMN_NAME] + new_anchor_data_column_names]
    dataframe.dropna(
        axis = 0,
        subset = [DEFAULT_TIMESTAMP_COLUMN_NAME, DEFAULT_OBJECT_ID_COLUMN_NAME],
        inplace = True
    )
    if min_data_value is not None:
        dataframe[new_anchor_data_column_names] = dataframe[new_anchor_data_column_names].mask(
            dataframe[new_anchor_data_column_names] <= min_data_value,
            np.nan
        )
    if max_data_value is not None:
        dataframe[new_anchor_data_column_names] = dataframe[new_anchor_data_column_names].mask(
            dataframe[new_anchor_data_column_names] >= max_data_value,
            np.nan
        )
    dataframe.dropna(
        axis = 0,
        how = 'all',
        subset = new_anchor_data_column_names,
        inplace = True
    )
    return dataframe

def add_ids_to_dataframe(
    dataframe,
    **kwargs
):
    for column_name, column_value in kwargs.items():
        dataframe[column_name] = column_value
    return dataframe

def filter_dataframe(
    dataframe,
    anchor_ids = None,
    object_ids = None,
    start_timestamp = None,
    end_timestamp = None
):
    if anchor_ids is None and object_ids is None and start_timestamp is None and end_timestamp is None:
        return dataframe
    if anchor_ids is not None:
        anchor_id_boolean = dataframe[DEFAULT_ANCHOR_ID_COLUMN_NAME].isin(anchor_ids)
    else:
        anchor_id_boolean = True
    if object_ids is not None:
        object_id_boolean = dataframe[DEFAULT_OBJECT_ID_COLUMN_NAME].isin(object_ids)
    else:
        object_id_boolean = True
    if start_timestamp is not None:
        start_timestamp_boolean = (dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME] >= start_timestamp)
    else:
        start_timestamp_boolean = True
    if end_timestamp is not None:
        end_timestamp_boolean = (dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME] <= end_timestamp)
    else:
        end_timestamp_boolean = True
    combined_boolean = anchor_id_boolean & object_id_boolean & start_timestamp_boolean & end_timestamp_boolean
    dataframe_filtered = dataframe[combined_boolean]
    return dataframe_filtered

def wide_filter_dataframe(
    dataframe,
    object_ids = None,
    start_timestamp = None,
    end_timestamp = None
):
    if object_ids is None and start_timestamp is None and end_timestamp is None:
        return dataframe
    if object_ids is not None:
        object_id_boolean = dataframe[DEFAULT_OBJECT_ID_COLUMN_NAME].isin(object_ids)
    else:
        object_id_boolean = True
    if start_timestamp is not None:
        start_timestamp_boolean = (dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME] >= start_timestamp)
    else:
        start_timestamp_boolean = True
    if end_timestamp is not None:
        end_timestamp_boolean = (dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME] <= end_timestamp)
    else:
        end_timestamp_boolean = True
    combined_boolean = object_id_boolean & start_timestamp_boolean & end_timestamp_boolean
    dataframe_filtered = dataframe[combined_boolean]
    return dataframe_filtered

def csv_directories_to_dataframe(
    top_directory,
    directory_filter = None,
    directory_parser = None,
    filename_filter = None,
    filename_parser = None,
    add_ids = {},
    anchor_ids = None,
    object_ids = None,
    start_timestamp = None,
    end_timestamp = None,
    timestamp_column_name = DEFAULT_TIMESTAMP_COLUMN_NAME,
    anchor_id_column_name = DEFAULT_ANCHOR_ID_COLUMN_NAME,
    object_id_column_name = DEFAULT_OBJECT_ID_COLUMN_NAME,
    measurement_value_column_name = DEFAULT_MEASUREMENT_VALUE_COLUMN_NAME
):
    dataframes = []
    directory_entries = os.listdir(top_directory)
    for directory_entry in directory_entries:
        path = os.path.join(top_directory, directory_entry)
        if os.path.isdir(path) and (directory_filter is None or directory_filter.match(directory_entry)):
            if directory_parser is not None:
                additional_ids = directory_parser(directory_entry)
                add_ids.update(additional_ids)
            dataframe = csv_files_to_dataframe(
                directory = path,
                filename_filter = filename_filter,
                filename_parser = filename_parser,
                add_ids = add_ids,
                anchor_ids = anchor_ids,
                object_ids = object_ids,
                start_timestamp = start_timestamp,
                end_timestamp = end_timestamp,
                timestamp_column_name = timestamp_column_name,
                anchor_id_column_name = anchor_id_column_name,
                object_id_column_name = object_id_column_name,
                measurement_value_column_name = measurement_value_column_name
            )
            if dataframe is not None and len(dataframe) > 0:
                print('Adding {} rows from directory {} spanning {} to {}'.format(
                    len(dataframe),
                    path,
                    dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME].min().isoformat(),
                    dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME].max().isoformat()))
                dataframes.append(dataframe)
    if len(dataframes) == 0:
        return None
    dataframe_all = pd.concat(dataframes, ignore_index = True)
    dataframe_all = dataframe_all[[
        DEFAULT_TIMESTAMP_COLUMN_NAME,
        DEFAULT_OBJECT_ID_COLUMN_NAME,
        DEFAULT_ANCHOR_ID_COLUMN_NAME,
        DEFAULT_MEASUREMENT_VALUE_COLUMN_NAME
    ]]
    dataframe_all.sort_values(DEFAULT_TIMESTAMP_COLUMN_NAME, inplace=True)
    dataframe_all.reset_index(inplace = True, drop = True)
    return dataframe_all

def wide_csv_directories_to_dataframe(
    top_directory,
    timestamp_column_name,
    anchor_data_column_names,
    object_id_column_name,
    directory_filter = None,
    directory_parser = None,
    filename_filter = None,
    filename_parser = None,
    add_ids = {},
    object_ids = None,
    start_timestamp = None,
    end_timestamp = None,
    min_data_value = None,
    max_data_value = None
):
    dataframes = []
    directory_entries = os.listdir(top_directory)
    for directory_entry in directory_entries:
        path = os.path.join(top_directory, directory_entry)
        if os.path.isdir(path) and (directory_filter is None or directory_filter.match(directory_entry)):
            if directory_parser is not None:
                additional_ids = directory_parser(directory_entry)
                add_ids.update(additional_ids)
            dataframe = wide_csv_files_to_dataframe(
                directory = path,
                timestamp_column_name = timestamp_column_name,
                anchor_data_column_names = anchor_data_column_names,
                object_id_column_name = object_id_column_name,
                filename_filter = filename_filter,
                filename_parser = filename_parser,
                add_ids = add_ids,
                object_ids = object_ids,
                start_timestamp = start_timestamp,
                end_timestamp = end_timestamp,
                min_data_value = min_data_value,
                max_data_value = max_data_value
            )
            if dataframe is not None and len(dataframe) > 0:
                print('Adding {} rows from directory {} spanning {} to {}'.format(
                    len(dataframe),
                    path,
                    dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME].min().isoformat(),
                    dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME].max().isoformat()))
                dataframes.append(dataframe)
    if len(dataframes) == 0:
        return None
    dataframe_all = pd.concat(dataframes, ignore_index = True)
    dataframe_all.sort_values(DEFAULT_TIMESTAMP_COLUMN_NAME, inplace=True)
    dataframe_all.reset_index(inplace = True, drop = True)
    return dataframe_all

def csv_files_to_dataframe(
    directory,
    filename_filter = None,
    filename_parser = None,
    add_ids = {},
    anchor_ids = None,
    object_ids = None,
    start_timestamp = None,
    end_timestamp = None,
    timestamp_column_name = DEFAULT_TIMESTAMP_COLUMN_NAME,
    anchor_id_column_name = DEFAULT_ANCHOR_ID_COLUMN_NAME,
    object_id_column_name = DEFAULT_OBJECT_ID_COLUMN_NAME,
    measurement_value_column_name = DEFAULT_MEASUREMENT_VALUE_COLUMN_NAME
):
    dataframes = []
    directory_entries = os.listdir(directory)
    for directory_entry in directory_entries:
        path = os.path.join(directory, directory_entry)
        if not os.path.isdir(path) and (filename_filter is None or filename_filter.match(directory_entry)):
            if filename_parser is not None:
                additional_ids = filename_parser(directory_entry)
                add_ids.update(additional_ids)
            dataframe = csv_file_to_dataframe(
                directory = directory,
                filename = directory_entry,
                timestamp_column_name = timestamp_column_name,
                anchor_id_column_name = anchor_id_column_name,
                object_id_column_name = object_id_column_name,
                measurement_value_column_name = measurement_value_column_name
            )
            dataframe = add_ids_to_dataframe(
                dataframe,
                **add_ids
            )
            dataframe = filter_dataframe(
                dataframe,
                anchor_ids,
                object_ids,
                start_timestamp,
                end_timestamp,
            )
            if len(dataframe) > 0:
                print('Adding {} rows from file {} spanning {} to {}'.format(
                    len(dataframe),
                    path,
                    dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME].min().isoformat(),
                    dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME].max().isoformat()))
                dataframes.append(dataframe)
    if len(dataframes) == 0:
         return None
    dataframe_all = pd.concat(dataframes, ignore_index = True)
    dataframe_all = dataframe_all[[
        DEFAULT_TIMESTAMP_COLUMN_NAME,
        DEFAULT_OBJECT_ID_COLUMN_NAME,
        DEFAULT_ANCHOR_ID_COLUMN_NAME,
        DEFAULT_MEASUREMENT_VALUE_COLUMN_NAME
    ]]
    dataframe_all.sort_values(DEFAULT_TIMESTAMP_COLUMN_NAME, inplace=True)
    dataframe_all.reset_index(inplace = True, drop = True)
    return dataframe_all

def wide_csv_files_to_dataframe(
    directory,
    timestamp_column_name,
    anchor_data_column_names,
    object_id_column_name,
    filename_filter = None,
    filename_parser = None,
    add_ids = {},
    object_ids = None,
    start_timestamp = None,
    end_timestamp = None,
    min_data_value = None,
    max_data_value = None
):
    dataframes = []
    directory_entries = os.listdir(directory)
    for directory_entry in directory_entries:
        path = os.path.join(directory, directory_entry)
        if not os.path.isdir(path) and (filename_filter is None or filename_filter.match(directory_entry)):
            if filename_parser is not None:
                additional_ids = filename_parser(directory_entry)
                add_ids.update(additional_ids)
            dataframe = wide_csv_file_to_dataframe(
                directory = directory,
                filename = directory_entry,
                timestamp_column_name = timestamp_column_name,
                anchor_data_column_names = anchor_data_column_names,
                object_id_column_name = object_id_column_name,
                min_data_value = min_data_value,
                max_data_value = max_data_value
            )
            dataframe = add_ids_to_dataframe(
                dataframe,
                **add_ids
            )
            dataframe = wide_filter_dataframe(
                dataframe,
                object_ids,
                start_timestamp,
                end_timestamp,
            )
            if len(dataframe) > 0:
                print('Adding {} rows from file {} spanning {} to {}'.format(
                    len(dataframe),
                    path,
                    dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME].min().isoformat(),
                    dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME].max().isoformat()))
                dataframes.append(dataframe)
    if len(dataframes) == 0:
         return None
    dataframe_all = pd.concat(dataframes, ignore_index = True)
    dataframe_all.sort_values(DEFAULT_TIMESTAMP_COLUMN_NAME, inplace=True)
    dataframe_all.reset_index(inplace = True, drop = True)
    return dataframe_all

def csv_directories_to_npz_files_by_object_one_day(
    input_top_directory,
    output_directory,
    output_filename_stem,
    measurement_value_name,
    year,
    month,
    day,
    start_hour = 12,
    end_hour = 19,
    tz = 'UTC',
    directory_filter = None,
    directory_parser = None,
    filename_filter = None,
    filename_parser = None,
    add_ids = {},
    timestamp_column_name = DEFAULT_TIMESTAMP_COLUMN_NAME,
    anchor_id_column_name = DEFAULT_ANCHOR_ID_COLUMN_NAME,
    object_id_column_name = DEFAULT_OBJECT_ID_COLUMN_NAME,
    measurement_value_column_name = DEFAULT_MEASUREMENT_VALUE_COLUMN_NAME
):
    start_timestamp = pd.Timestamp(year = year, month = month, day = day, hour = start_hour, tz=tz)
    end_timestamp = pd.Timestamp(year = year, month = month, day = day, hour = end_hour, tz=tz)
    print('Retrieving observations between {} and {}'.format(
        start_timestamp.astimezone('UTC').isoformat(),
        end_timestamp.astimezone('UTC').isoformat(),
    ))
    dataframe_all = csv_directories_to_dataframe(
        top_directory = input_top_directory,
        directory_filter = directory_filter,
        directory_parser = directory_parser,
        filename_filter = filename_filter,
        filename_parser = filename_parser,
        add_ids = add_ids,
        anchor_ids = None,
        object_ids = None,
        start_timestamp = start_timestamp,
        end_timestamp = end_timestamp,
        timestamp_column_name = timestamp_column_name,
        anchor_id_column_name = anchor_id_column_name,
        object_id_column_name = object_id_column_name,
        measurement_value_column_name = measurement_value_column_name
    )
    print('Gathered {} observations'.format(len(dataframe_all)))
    arrays_by_object = dataframe_to_arrays_by_object(
        dataframe = dataframe_all,
        measurement_value_name = measurement_value_name)
    output_filename_stem_with_date = '{}_{:04}{:02}{:02}'.format(output_filename_stem, year, month, day)
    arrays_by_object_to_npz_files_by_object(
        arrays_by_object,
        directory = output_directory,
        filename_stem = output_filename_stem_with_date
    )

def dataframe_to_arrays(dataframe, measurement_value_name):
    timestamps = np.sort(dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME].unique())
    anchor_ids = np.sort(dataframe[DEFAULT_ANCHOR_ID_COLUMN_NAME].unique())
    object_ids = np.sort(dataframe[DEFAULT_OBJECT_ID_COLUMN_NAME].unique())
    num_timestamps = len(timestamps)
    num_anchors = len(anchor_ids)
    num_objects = len(object_ids)
    dataframe_all = pd.DataFrame(
        list(itertools.product(
            timestamps,
            anchor_ids,
            object_ids
        )),
        columns = [
            DEFAULT_TIMESTAMP_COLUMN_NAME,
            DEFAULT_ANCHOR_ID_COLUMN_NAME,
            DEFAULT_OBJECT_ID_COLUMN_NAME
        ]
    )
    dataframe_merged = dataframe_all.merge(
        right = dataframe,
        how = 'left',
        on = [
            DEFAULT_TIMESTAMP_COLUMN_NAME,
            DEFAULT_ANCHOR_ID_COLUMN_NAME,
            DEFAULT_OBJECT_ID_COLUMN_NAME
        ]
    )
    measurement_values = dataframe_merged[DEFAULT_MEASUREMENT_VALUE_COLUMN_NAME].values
    measurement_value_array = measurement_values.reshape(
        num_timestamps,
        1,
        num_anchors,
        num_objects
    )
    arrays = {
        'num_timestamps': num_timestamps,
        'num_anchors': num_anchors,
        'num_objects': num_objects,
        'timestamps': timestamps,
        'anchor_ids': anchor_ids,
        'object_ids': object_ids,
        measurement_value_name: measurement_value_array
    }
    return arrays

def wide_dataframe_to_arrays(dataframe, num_anchors, measurement_value_name):
    timestamps = np.sort(dataframe[DEFAULT_TIMESTAMP_COLUMN_NAME].unique())
    object_ids = np.sort(dataframe[DEFAULT_OBJECT_ID_COLUMN_NAME].unique())
    num_timestamps = len(timestamps)
    num_objects = len(object_ids)
    dataframe_all = pd.DataFrame(
        list(itertools.product(
            timestamps,
            object_ids
        )),
        columns = [
            DEFAULT_TIMESTAMP_COLUMN_NAME,
            DEFAULT_OBJECT_ID_COLUMN_NAME
        ]
    )
    dataframe_merged = dataframe_all.merge(
        right = dataframe,
        how = 'left',
        on = [
            DEFAULT_TIMESTAMP_COLUMN_NAME,
            DEFAULT_OBJECT_ID_COLUMN_NAME
        ]
    )
    anchor_data_column_names = ['{}{:02}'.format(DEFAULT_ANCHOR_DATA_COLUMN_NAME_PREFIX, i) for i in range(num_anchors)]
    measurement_values = dataframe_merged[anchor_data_column_names].values
    measurement_value_array = measurement_values.reshape(
        num_timestamps,
        1,
        num_objects,
        num_anchors
    )
    measurement_value_array = np.swapaxes(measurement_value_array, 2, 3)
    arrays = {
        'num_timestamps': num_timestamps,
        'num_anchors': num_anchors,
        'num_objects': num_objects,
        'timestamps': timestamps,
        'anchor_ids': anchor_data_column_names,
        'object_ids': object_ids,
        measurement_value_name: measurement_value_array
    }
    return arrays

def dataframe_to_arrays_by_object(dataframe, measurement_value_name):
    arrays_dict = {}
    for group_name, dataframe_single_object in dataframe.groupby(DEFAULT_OBJECT_ID_COLUMN_NAME):
        object_id = group_name
        print('Processing data for object {} ({} rows spanning {} to {})...'.format(
            object_id,
            len(dataframe_single_object),
            dataframe_single_object[DEFAULT_TIMESTAMP_COLUMN_NAME].min().isoformat(),
            dataframe_single_object[DEFAULT_TIMESTAMP_COLUMN_NAME].max().isoformat()))
        arrays_dict[object_id] = dataframe_to_arrays(
            dataframe = dataframe_single_object,
            measurement_value_name = measurement_value_name
        )
    return arrays_dict

def arrays_to_observation_database(arrays, measurement_value_name):
    observation_structure = smcmodel_localize.model.observation_structure_generator(arrays['num_anchors'], arrays['num_objects'], measurement_value_name)
    observation_time_series_data = {measurement_value_name: arrays[measurement_value_name]}
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
    filename_stem
):
    for object_id, dataframe_single_object in dataframe.groupby(DEFAULT_OBJECT_ID_COLUMN_NAME):
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
    filename,
    measurement_value_name
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
        measurement_value_name: np.asarray(npz_data[measurement_value_name])
    }
    return data

def get_object_info_from_csv_file(
    object_ids,
    directory,
    filename,
    fixed_object_positions_column_names,
    object_name_column_name = None,
    object_id_column_name = DEFAULT_OBJECT_ID_COLUMN_NAME
):
    object_info_dataframe = pd.DataFrame.from_dict({DEFAULT_OBJECT_ID_COLUMN_NAME: object_ids})
    path = os.path.join(directory, filename)
    file_dataframe = pd.read_csv(path)
    object_info_dataframe = object_info_dataframe.merge(
        right = file_dataframe,
        how = 'left',
        left_on = DEFAULT_OBJECT_ID_COLUMN_NAME,
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
    anchor_id_column_name = DEFAULT_ANCHOR_ID_COLUMN_NAME
):
    anchor_info_dataframe = pd.DataFrame.from_dict({DEFAULT_ANCHOR_ID_COLUMN_NAME: anchor_ids})
    path = os.path.join(directory, filename)
    file_dataframe = pd.read_csv(path)
    anchor_info_dataframe = anchor_info_dataframe.merge(
        right = file_dataframe,
        how = 'left',
        left_on = DEFAULT_ANCHOR_ID_COLUMN_NAME,
        right_on = anchor_id_column_name)
    anchor_positions = anchor_info_dataframe[anchor_positions_column_names].values
    anchor_info = {
        'anchor_positions': anchor_positions
    }
    if anchor_name_column_name is not None:
        anchor_names = anchor_info_dataframe[anchor_name_column_name].values.tolist()
        anchor_info['anchor_names'] = anchor_names
    return anchor_info
