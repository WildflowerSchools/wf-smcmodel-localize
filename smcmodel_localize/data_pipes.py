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
