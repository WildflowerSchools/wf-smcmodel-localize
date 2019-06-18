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
    state_summary_timestamps_np = datetime_conversion.to_numpy_datetimes(state_summary_timestamps)
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

def plot_positions_topdown(
    state_summary_database,
    start_timestamp = None,
    end_timestamp = None,
    object_ids = None,
    object_names = None,
    position_axes_names = ['$x$', '$y$'],
    timezone_name = 'UTC',
    output_path = None):
    state_summary_timestamps, state_summary_time_series = state_summary_database.fetch_data(
        start_timestamp = start_timestamp,
        end_timestamp = end_timestamp)
    num_objects = state_summary_time_series['moving_object_positions_mean'].shape[2]
    state_summary_timestamps_np = datetime_conversion.to_numpy_datetimes(state_summary_timestamps)
    date_formatter = mdates.DateFormatter('%H:%M')
    for object_index in range(num_objects):
        if object_names is not None:
            title_object_name = object_names[object_index]
        elif object_ids is not None:
            title_object_name = object_ids[object_index]
        else:
            title_object_name = 'Object {}'.format(object_index)
        fig, ax = plt.subplots()
        plt.plot(
            state_summary_time_series['moving_object_positions_mean'][:, 0, object_index, 0],
            state_summary_time_series['moving_object_positions_mean'][:, 0, object_index, 1],
            color='blue',
            label = 'Mean estimate'
        )
        lgd = plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('{} position'.format(position_axes_names[0]))
        plt.ylabel('{} position'.format(position_axes_names[1]))
        plt.title('Sensor: {}'.format(title_object_name))
        ax.set_aspect('equal')
        if output_path is not None:
            plt.savefig(output_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()

def plot_state_summary_timestamp_density(
    state_summary_database,
    start_timestamp = None,
    end_timestamp = None,
    bins = 100,
    timezone_name = 'UTC'):
    state_summary_timestamps, state_summary_time_series = state_summary_database.fetch_data(
        start_timestamp = start_timestamp,
        end_timestamp = end_timestamp)
    state_summary_timestamps_np = datetime_conversion.to_numpy_datetimes(state_summary_timestamps)
    date_formatter = mdates.DateFormatter('%H:%M')
    fig, ax = plt.subplots()
    plt.hist(
        state_summary_timestamps_np,
        bins = bins,
        color = 'blue'
    )
    plt.xlabel('Time ({})'.format(timezone_name))
    plt.ylabel('Number of observations')
    ax.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()
    plt.show()

def plot_num_samples(
    state_summary_database,
    start_timestamp = None,
    end_timestamp = None,
    timezone_name = 'UTC'):
    state_summary_timestamps, state_summary_time_series = state_summary_database.fetch_data(
        start_timestamp = start_timestamp,
        end_timestamp = end_timestamp)
    state_summary_timestamps_np = datetime_conversion.to_numpy_datetimes(state_summary_timestamps)
    date_formatter = mdates.DateFormatter('%H:%M')
    fig, ax = plt.subplots()
    plt.plot(
        state_summary_timestamps_np[:],
        state_summary_time_series['num_resample_indices'][:, 0],
        color='blue'
    )
    plt.xlabel('Time ({})'.format(timezone_name))
    plt.ylabel('Number of samples')
    plt.title('Number of samples at each time step')
    ax.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()
    plt.show()
