import datetime_conversion
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
import os

register_matplotlib_converters()

def plot_positions(
    state_summary_data_destination,
    # state_summary_database,
    start_timestamp = None,
    end_timestamp = None,
    title_object_names = None,
    title_addendum = None,
    position_axes_names = ['$x$', '$y$'],
    timezone_name = 'UTC',
    x_size_inches = 7.5,
    y_size_inches = 10,
    save = True,
    output_directory = '.',
    output_filename_stem = None,
    output_filename_object_ids = None,
    output_filename_extension = 'png',
    show = False
):
    state_summary_timestamps = state_summary_data_destination.timestamps
    state_summary_time_series = state_summary_data_destination.array_dict
    # state_summary_timestamps, state_summary_time_series = state_summary_database.fetch_data(
    #     start_timestamp = start_timestamp,
    #     end_timestamp = end_timestamp)
    num_objects = state_summary_time_series['moving_object_positions_mean'].shape[2]
    num_position_axes = state_summary_time_series['moving_object_positions_mean'].shape[3]
    state_summary_timestamps_np = datetime_conversion.to_numpy_datetimes(state_summary_timestamps)
    date_formatter = mdates.DateFormatter('%H:%M')
    for object_index in range(num_objects):
        if title_object_names is not None:
            title_string = title_object_names[object_index]
        else:
            title_string = 'Object {}'.format(object_index)
        if title_addendum is not None:
            title_string += ' ({})'.format(title_addendum)
        fig, axes = plt.subplots(nrows = num_position_axes, ncols = 1, sharex = True)
        for position_axis_index in range(num_position_axes):
            axes[position_axis_index].plot(
                state_summary_timestamps_np[:],
                state_summary_time_series['moving_object_positions_mean'][:, 0, object_index, position_axis_index],
                color='blue',
                label = 'Mean estimate'
            )
            lgd = axes[position_axis_index].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            axes[position_axis_index].set_xlabel('Time ({})'.format(timezone_name))
            axes[position_axis_index].set_ylabel('{} position'.format(position_axes_names[position_axis_index]))
            axes[position_axis_index].xaxis.set_major_formatter(date_formatter)
        fig_suptitle = fig.suptitle(title_string, fontsize = 'x-large')
        fig.autofmt_xdate()
        fig.set_size_inches(x_size_inches, y_size_inches)
        if save:
            if output_filename_object_ids is not None:
                output_filename_object_id = output_filename_object_ids[object_index]
            else:
                output_filename_object_id = 'obj{:02}'.format(object_index)
            output_path = os.path.join(
                output_directory,
                'positions_{}_{}.{}'.format(
                    output_filename_stem,
                    output_filename_object_id,
                    output_filename_extension
                )
            )
            plt.savefig(output_path, bbox_extra_artists=(lgd, fig_suptitle), bbox_inches='tight')
        if show:
            plt.show()

def plot_positions_topdown(
    state_summary_data_destination,
    # state_summary_database,
    start_timestamp = None,
    end_timestamp = None,
    title_object_names = None,
    title_addendum = None,
    position_axes_names = ['$x$', '$y$'],
    x_size_inches = 7.5,
    y_size_inches = 10,
    save = True,
    output_directory = '.',
    output_filename_stem = None,
    output_filename_object_ids = None,
    output_filename_extension = 'png',
    show = False
    ):
    state_summary_timestamps = state_summary_data_destination.timestamps
    state_summary_time_series = state_summary_data_destination.array_dict
    # state_summary_timestamps, state_summary_time_series = state_summary_database.fetch_data(
    #     start_timestamp = start_timestamp,
    #     end_timestamp = end_timestamp)
    num_objects = state_summary_time_series['moving_object_positions_mean'].shape[2]
    state_summary_timestamps_np = datetime_conversion.to_numpy_datetimes(state_summary_timestamps)
    for object_index in range(num_objects):
        if title_object_names is not None:
            title_string = title_object_names[object_index]
        else:
            title_string = 'Object {}'.format(object_index)
        if title_addendum is not None:
            title_string += ' ({})'.format(title_addendum)
        fig, ax = plt.subplots()
        ax.plot(
            state_summary_time_series['moving_object_positions_mean'][:, 0, object_index, 0],
            state_summary_time_series['moving_object_positions_mean'][:, 0, object_index, 1],
            color='blue',
            label = 'Mean estimate'
        )
        lgd = ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        ax.set_xlabel('{} position'.format(position_axes_names[0]))
        ax.set_ylabel('{} position'.format(position_axes_names[1]))
        ax.set_aspect('equal')
        fig_suptitle = fig.suptitle(title_string, fontsize = 'x-large')
        fig.set_size_inches(x_size_inches, y_size_inches)
        if save:
            if output_filename_object_ids is not None:
                output_filename_object_id = output_filename_object_ids[object_index]
            else:
                output_filename_object_id = 'obj{:02}'.format(object_index)
            output_path = os.path.join(
                output_directory,
                'positions_topdown_{}_{}.{}'.format(
                    output_filename_stem,
                    output_filename_object_id,
                    output_filename_extension
                )
            )
            plt.savefig(output_path, bbox_extra_artists=(lgd, fig_suptitle), bbox_inches='tight')
        if show:
            plt.show()

def plot_state_summary_timestamp_density(
    state_summary_data_destination,
    # state_summary_database,
    start_timestamp = None,
    end_timestamp = None,
    bins = 100,
    timezone_name = 'UTC',
    fig_title = 'Timestamp density',
    x_size_inches = 7.5,
    y_size_inches = 6,
    save = True,
    output_directory = '.',
    output_filename_stem = None,
    output_filename_extension = 'png',
    show = False
    ):
    state_summary_timestamps = state_summary_data_destination.timestamps
    state_summary_time_series = state_summary_data_destination.array_dict
    # state_summary_timestamps, state_summary_time_series = state_summary_database.fetch_data(
    #     start_timestamp = start_timestamp,
    #     end_timestamp = end_timestamp)
    state_summary_timestamps_np = datetime_conversion.to_numpy_datetimes(state_summary_timestamps)
    date_formatter = mdates.DateFormatter('%H:%M')
    fig, ax = plt.subplots()
    ax.hist(
        state_summary_timestamps_np,
        bins = bins,
        color = 'blue'
    )
    ax.set_xlabel('Time ({})'.format(timezone_name))
    ax.set_ylabel('Number of observations')
    ax.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()
    fig_suptitle = fig.suptitle(fig_title, fontsize = 'x-large')
    fig.set_size_inches(x_size_inches, y_size_inches)
    if save:
        output_path = os.path.join(
            output_directory,
            'timestamp_density_{}.{}'.format(
                output_filename_stem,
                output_filename_extension
            )
        )
        plt.savefig(output_path, bbox_extra_artists=(fig_suptitle,), bbox_inches='tight')
    if show:
        plt.show()

def plot_num_samples(
    state_summary_data_destination,
    # state_summary_database,
    start_timestamp = None,
    end_timestamp = None,
    timezone_name = 'UTC',
    fig_title = 'Number of samples after resampling',
    x_size_inches = 7.5,
    y_size_inches = 6,
    save = True,
    output_directory = '.',
    output_filename_stem = None,
    output_filename_extension = 'png',
    show = False
    ):
    state_summary_timestamps = state_summary_data_destination.timestamps
    state_summary_time_series = state_summary_data_destination.array_dict
    # state_summary_timestamps, state_summary_time_series = state_summary_database.fetch_data(
    #     start_timestamp = start_timestamp,
    #     end_timestamp = end_timestamp)
    state_summary_timestamps_np = datetime_conversion.to_numpy_datetimes(state_summary_timestamps)
    date_formatter = mdates.DateFormatter('%H:%M')
    fig, ax = plt.subplots()
    ax.plot(
        state_summary_timestamps_np[:],
        state_summary_time_series['num_resample_indices'][:, 0],
        color='blue'
    )
    ax.set_xlabel('Time ({})'.format(timezone_name))
    ax.set_ylabel('Number of samples')
    ax.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()
    fig_suptitle = fig.suptitle(fig_title, fontsize = 'x-large')
    fig.set_size_inches(x_size_inches, y_size_inches)
    if save:
        output_path = os.path.join(
            output_directory,
            'num_samples_{}.{}'.format(
                output_filename_stem,
                output_filename_extension
            )
        )
        plt.savefig(output_path, bbox_extra_artists=(fig_suptitle,), bbox_inches='tight')
    if show:
        plt.show()
