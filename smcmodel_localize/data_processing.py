import pandas as pd
import numpy as np
import time

def parse_rssi_dataframe(df, timestamp_column, anchor_id_column, object_id_column, rssi_column):
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
    return {
        'num_timestamps': num_timestamps,
        'num_anchors': num_anchors,
        'num_objects': num_objects,
        'timestamps': timestamps,
        'anchor_ids': anchor_ids,
        'object_ids': object_ids,
        'rssis': rssis
    }

def parse_rssi_dataframe_separate_objects(df, timestamp_column, anchor_id_column, object_id_column, rssi_column):
    rssi_arrays_dict = {}
    for group_name, df_single_object in df.groupby(object_id_column):
        object_id = group_name
        print('Processing data for object {} ({} rows)...'.format(object_id, len(df_single_object)))
        rssi_arrays_dict[object_id] = parse_rssi_dataframe(
            df_single_object,
            timestamp_column,
            anchor_id_column,
            object_id_column,
            rssi_column)
    return rssi_arrays_dict
