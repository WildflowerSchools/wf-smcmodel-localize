import smcmodel_localize.model
import numpy as np
import scipy.optimize
import tqdm

class LocalizationModelMultilateration:

    def __init__(
        self,
        num_objects,
        num_anchors,
        anchor_positions,
        initial_position_guesses,
        num_moving_object_dimensions = 3,
        num_fixed_object_dimensions = 0,
        fixed_object_positions = None,
        measurement_value_name = 'range'
    ):
        num_object_dimensions = num_moving_object_dimensions + num_fixed_object_dimensions
        anchor_positions = np.asarray(anchor_positions)
        initial_position_guesses = np.asarray(initial_position_guesses)
        if anchor_positions.shape[0] != num_anchors:
            raise ValueError('Number of anchors specified as {} but positions specified for {} anchors'.format(
                num_anchors,
                anchor_positions.shape[0]
            ))
        if anchor_positions.shape[1] != num_object_dimensions:
            raise ValueError('Specified anchor positions have {} spatial dimensions but model is specified to have {} + {} = {} object dimensions'.format(
                anchor_positions.shape[1],
                num_moving_object_dimensions,
                num_fixed_object_dimensions,
                num_object_dimensions
            ))
        if initial_position_guesses.shape[0] != num_objects:
            raise ValueError('Number of objects specified as {} but initial position guesses specified for {} objects'.format(
                num_objects,
                initial_position_guesses.shape[0]
            ))
        if initial_position_guesses.shape[1] != num_moving_object_dimensions:
            raise ValueError('Number of moving object dimensions specified as {} but initial position guesses have {} spatial dimensions'.format(
                num_moving_object_dimensions,
                initial_position_guesses.shape[1]
            ))
        if num_fixed_object_dimensions != 0 and fixed_object_positions is None:
            raise ValueError('Number of fixed object dimensions specified as {} but no fixed object positions specified')
        if fixed_object_positions is not None:
            fixed_object_positions = np.asarray(fixed_object_positions)
            if fixed_object_positions.shape[0] != num_objects:
                raise ValueError('Number of objects specified as {} but fixed object object positions specified for {} objects'.format(
                    num_objects,
                    fixed_object_positions.shape[0]
                ))
            if fixed_object_positions.shape[1] != num_fixed_object_dimensions:
                raise ValueError('Number of fixed object dimensions specified as {} but specified fixed object positions have {} spatial dimensions'.format(
                    num_fixed_object_dimensions,
                    fixed_object_positions.shape[1]
                ))
        self.num_objects = num_objects
        self.num_anchors = num_anchors
        self.anchor_positions = anchor_positions
        self.initial_position_guesses = initial_position_guesses
        self.num_moving_object_dimensions = num_moving_object_dimensions
        self.num_fixed_object_dimensions = num_fixed_object_dimensions
        self.num_object_dimensions = num_object_dimensions
        self.fixed_object_positions = fixed_object_positions
        self.measurement_value_name = measurement_value_name
        self.observation_structure = smcmodel_localize.model.observation_structure_generator(
            self.num_anchors,
            self.num_objects,
            self.measurement_value_name
        )
        self.state_summary_structure = state_summary_structure_generator_multilateration(
            self.num_objects,
            self.num_moving_object_dimensions
        )

    def estimate_state_time_series_multilateration(
        self,
        observation_data_queue,
        state_summary_database,
        progress_bar = False,
        num_observations = None
    ):
        position_guesses = self.initial_position_guesses
        if progress_bar:
            if num_observations is not None:
                remaining_iterations = num_observations
            else:
                remaining_iterations = None
            t = tqdm.tqdm(total = remaining_iterations)
        for timestamp, observation_arrays in observation_data_queue:
            observation = observation_arrays[self.measurement_value_name]
            if observation.shape[1] != self.num_anchors:
                raise ValueError('Number of anchors specified as {} but observation data contains ranges for {} anchors'.format(
                    self.num_anchors,
                    observation.shape[1],
                ))
            if observation.shape[2] != self.num_objects:
                raise ValueError('Number of objects specified as {} but observation data contains ranges for {} objects'.format(
                    self.num_objects,
                    observation.shape[2],
                ))
            if position_guesses.shape[0] != self.num_objects:
                raise ValueError('Number of objects specified as {} but initial position guesses specified for {} objects'.format(
                    self.num_objects,
                    position_guesses.shape[0]
                ))
            if position_guesses.shape[1] != self.num_moving_object_dimensions:
                raise ValueError('Number of moving object dimensions specified as {} but initial position guesses have {} spatial dimensions'.format(
                    self.num_moving_object_dimensions,
                    position_guesses.shape[1]
                ))
            if self.fixed_object_positions is not None:
                initial_guesses = np.concatenate(
                    (position_guesses, self.fixed_object_positions),
                    axis = -1
                )
            else:
                initial_guesses = position_guesses
            moving_object_positions = np.empty((self.num_objects, self.num_object_dimensions))
            for object_index in range(self.num_objects):
                moving_object_positions[object_index] = self.estimated_position(
                    observation[0, :, object_index],
                    position_guesses[object_index]
                )
            moving_object_positions_expanded = np.expand_dims(moving_object_positions, axis = 0)
            state_summary = {
                'moving_object_positions_multilateration': moving_object_positions_expanded,
            }
            state_summary_database.write_data(
                timestamp, state_summary)
            position_guesses = moving_object_positions[:, :self.num_moving_object_dimensions]
            if progress_bar:
                t.update()
        if progress_bar:
            t.close()

    def estimated_position(self,measured_ranges, initial_guess):
        solution = scipy.optimize.minimize(
            fun = self.mean_squared_error,
            x0 = initial_guess,
            args = (measured_ranges))
        return solution.x

    def mean_squared_error(self, object_position, measured_ranges):
        num_measured_ranges = np.sum(np.logical_not(np.isnan(measured_ranges)))
        calculated_ranges = self.ranges(object_position)
        errors = measured_ranges - calculated_ranges
        sum_squared_errors = np.nansum(np.square(errors))
        mean_squared_error = sum_squared_errors/num_measured_ranges
        return mean_squared_error

    def ranges(self, object_position):
        object_position_expanded = np.expand_dims(object_position, 0)
        return np.linalg.norm(object_position_expanded - self.anchor_positions, axis = -1)

def state_summary_structure_generator_multilateration(
    num_objects,
    num_moving_object_dimensions
):
    state_summary_structure = {
        'moving_object_positions_multilateration': {
            'shape': [num_objects, num_moving_object_dimensions],
            'type': 'float32'
        }
    }
    return state_summary_structure
