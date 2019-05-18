import datetime_conversion
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

def plot_positions(
    state_summary_database,
    start_timestamp = None,
    end_timestamp = None,
    object_ids = None,
    object_names = None,
    position_axes_names = ['$x$', '$y$'],
    timezone_name = 'UTC'):
    state_summary_timestamps, state_summary_time_series = state_summary_database.fetch_data(
        start_timestamp = start_timestamp,
        end_timestamp = end_timestamp)
    num_objects = state_summary_time_series['moving_object_positions_mean'].shape[2]
    print(state_summary_timestamps)
    state_summary_timestamps_np = datetime_conversion.to_numpy_datetimes(state_summary_timestamps)
    print(state_summary_timestamps_np)
    date_formatter = mdates.DateFormatter('%H:%M')
    for object_index in range(num_objects):
        if object_names is not None:
            title_object_name = object_names[object_index]
        elif object_ids is not None:
            title_object_name = object_ids[object_index]
        else:
            title_object_name = 'Object {}'.format(object_index)
        for position_axis_index, position_axis_name in enumerate(position_axes_names):
            fig, ax = plt.subplots()
            plt.plot(
                state_summary_timestamps_np[:],
                state_summary_time_series['moving_object_positions_mean'][:, 0, object_index, position_axis_index],
                color='blue',
                label = 'Mean estimate'
            )
            plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.xlabel('Time ({})'.format(timezone_name))
            plt.ylabel('{} position'.format(position_axis_name))
            plt.title('Sensor: {}'.format(title_object_name))
            ax.xaxis.set_major_formatter(date_formatter)
            fig.autofmt_xdate()
            plt.show()
