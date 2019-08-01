import smcmodel.data_pipes
import smcmodel_localize.model
import pandas as pd
import numpy as np
import itertools

def list_to_df(data_list):
    df = pd.DataFrame(data_list)
    return df

def df_to_arrays(
    dataframe,
    measurement_value_field_name,
    timestamp_field_name = 'timestamp',
    object_id_field_name = 'object_id',
    anchor_id_field_name = 'anchor_id'
):
    timestamps = np.sort(dataframe[timestamp_field_name].unique())
    anchor_ids = np.sort(dataframe[anchor_id_field_name].unique())
    object_ids = np.sort(dataframe[object_id_field_name].unique())
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
            timestamp_field_name,
            anchor_id_field_name,
            object_id_field_name
        ]
    )
    dataframe_merged = dataframe_all.merge(
        right = dataframe,
        how = 'left',
        on = [
            timestamp_field_name,
            anchor_id_field_name,
            object_id_field_name
        ]
    )
    measurement_values = dataframe_merged[measurement_value_field_name].values
    measurement_value_array = measurement_values.reshape(
        num_timestamps,
        1,
        num_anchors,
        num_objects
    )
    timestamps_output = [timestamp.timestamp() for timestamp in timestamps.tolist()]
    anchor_ids_output = anchor_ids.tolist()
    object_ids_output = object_ids.tolist()
    arrays = {
        'timestamps': timestamps_output,
        'anchor_ids': anchor_ids_output,
        'object_ids': object_ids_output,
        measurement_value_field_name: measurement_value_array
    }
    return arrays

def arrays_to_observation_data_source(arrays, measurement_value_field_name):
    structure = smcmodel_localize.model.observation_structure_generator(
        num_anchors = len(arrays['anchor_ids']),
        num_objects = len(arrays['object_ids']),
        measurement_value_name = measurement_value_field_name
    )
    data_source = smcmodel.data_pipes.DataSourceArrayDict(
        structure = structure,
        num_samples = 1,
        timestamps = arrays['timestamps'],
        array_dict = arrays)
    return data_source

def get_object_info_from_csv_file(
    object_ids,
    path,
    object_id_column_name,
    fixed_object_positions_column_names,
    object_name_column_name = None
):
    object_info_dataframe = pd.DataFrame.from_dict({object_id_column_name: object_ids})
    file_dataframe = pd.read_csv(path)
    object_info_dataframe = object_info_dataframe.merge(
        right = file_dataframe,
        how = 'left',
        left_on = object_id_column_name,
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
    path,
    anchor_id_column_name,
    anchor_positions_column_names,
    anchor_name_column_name = None
):
    anchor_info_dataframe = pd.DataFrame.from_dict({anchor_id_column_name: anchor_ids})
    file_dataframe = pd.read_csv(path)
    anchor_info_dataframe = anchor_info_dataframe.merge(
        right = file_dataframe,
        how = 'left',
        left_on = anchor_id_column_name,
        right_on = anchor_id_column_name)
    anchor_positions = anchor_info_dataframe[anchor_positions_column_names].values
    anchor_info = {
        'anchor_positions': anchor_positions
    }
    if anchor_name_column_name is not None:
        anchor_names = anchor_info_dataframe[anchor_name_column_name].values.tolist()
        anchor_info['anchor_names'] = anchor_names
    return anchor_info

def create_state_summary_data_destination(num_objects, num_moving_object_dimensions):
    structure = smcmodel_localize.model.state_summary_structure_generator(num_objects, num_moving_object_dimensions)
    state_summary_data_destination = smcmodel.data_pipes.DataDestinationArrayDict(
        structure = structure,
        num_samples = 1
    )
    return state_summary_data_destination

def state_summary_data_destination_to_arrays(
    state_summary_data_destination
):
    arrays = {
        'timestamps': state_summary_data_destination.timestamps
    }
    arrays.update(state_summary_data_destination.array_dict)
    return arrays
