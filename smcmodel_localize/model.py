from smcmodel import SMCModelGeneralTensorflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class LocalizationModel(SMCModelGeneralTensorflow):

    def __init__(
        self,
        num_objects = 3,
        num_anchors = 4,
        num_moving_object_dimensions = 2,
        num_fixed_object_dimensions = 0,
        fixed_object_positions = None,
        room_corners = [[0.0, 0.0], [10.0, 20.0]],
        anchor_positions = [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 20.0],
            [10.0, 20.0]
        ],
        reference_time_interval = 1.0,
        reference_drift = 0.1,
        minimum_drift = 0.0001,
        measurement_value_mean_function = lambda x: x,
        measurement_value_sd_function = lambda x: reference_drift,
        measurement_value_name = 'measurement_values',
        ping_success_rate = 1.0
    ):
        if fixed_object_positions is not None and num_fixed_object_dimensions == 0:
            raise ValueError('If fixed_object_positions argument is present, num_fixed_object_dimensions argument must be > 0')
        if fixed_object_positions is None and num_fixed_object_dimensions != 0:
            raise ValueError('If num_fixed_object_dimensions argument > 0, fixed_object_positions argument must be present')
        if fixed_object_positions is not None:
            fixed_object_positions = np.asarray(fixed_object_positions)
            if fixed_object_positions.size != num_objects * num_fixed_object_dimensions:
                raise ValueError('If present, fixed_object_positions argument needs to be of size num_objects*num_fixed_object_dimensions')
            fixed_object_positions = np.reshape(fixed_object_positions, (num_objects, num_fixed_object_dimensions))
        else:
            fixed_object_positions = np.full((num_objects, num_fixed_object_dimensions), np.nan)
        room_corners = np.asarray(room_corners)
        if room_corners.shape != (2, num_moving_object_dimensions):
            raise ValueError('room_corners argument must be of shape (2, num_moving_object_dimensions)')
        num_dimensions = num_moving_object_dimensions + num_fixed_object_dimensions
        anchor_positions = np.asarray(anchor_positions)
        if anchor_positions.shape != (num_anchors, num_dimensions):
            raise ValueError('anchor_positions argument must be of shape (num_anchors, num_moving_object_dimensions + num_fixed_object_dimensions)')
        self.num_objects = num_objects
        self.num_anchors = num_anchors
        self.num_moving_object_dimensions = num_moving_object_dimensions
        self.num_fixed_object_dimensions = num_fixed_object_dimensions
        self.fixed_object_positions = fixed_object_positions
        self.room_corners = room_corners
        self.anchor_positions = anchor_positions
        self.reference_time_interval = reference_time_interval
        self.reference_drift = reference_drift
        self.minimum_drift = minimum_drift
        self.measurement_value_mean_function = measurement_value_mean_function
        self.measurement_value_sd_function = measurement_value_sd_function
        self.measurement_value_name = measurement_value_name
        self.ping_success_rate = ping_success_rate
        self.parameter_structure = parameter_structure_generator(
            self.num_anchors,
            self.num_objects,
            self.num_moving_object_dimensions,
            self.num_fixed_object_dimensions
        )
        self.state_structure = state_structure_generator(
            self.num_objects,
            self.num_moving_object_dimensions
        )
        self.observation_structure = observation_structure_generator(
            self.num_anchors,
            self.num_objects,
            self.measurement_value_name
        )
        self.state_summary_structure = state_summary_structure_generator(
            self.num_objects,
            self.num_moving_object_dimensions
        )
        if num_fixed_object_dimensions == 0:
            self.object_positions_function = self.object_positions_no_fixed_dimensions
        else:
            self.object_positions_function = self.object_positions_with_fixed_dimensions
        if ping_success_rate == 1.0:
            self.observation_model_sample = self.observation_model_sample_all_successful
        elif ping_success_rate == 0.0:
            self.observation_model_sample = self.observation_model_sample_one_successful
        elif ping_success_rate > 0.0 and ping_success_rate < 1.0:
            self.observation_model_sample = self.observation_model_sample_some_successful
        else:
            raise ValueError('Ping success rate out of range')

    def parameter_model_sample(self):
        parameters = {
            'fixed_object_positions': tf.constant(self.fixed_object_positions, dtype=tf.float32),
            'anchor_positions': tf.constant(self.anchor_positions, dtype = tf.float32),
            'reference_time_interval': tf.constant(self.reference_time_interval, dtype=tf.float32),
            'reference_drift': tf.constant(self.reference_drift, dtype=tf.float32),
            'ping_success_rate': tf.constant(self.ping_success_rate, dtype=tf.float32)
        }
        return parameters

    def initial_model_sample(self, num_samples, parameters):
        num_objects = self.num_objects
        room_corners = tf.constant(self.room_corners, dtype=tf.float32)
        room_distribution = tfp.distributions.Uniform(
            low = room_corners[0],
            high= room_corners[1]
        )
        initial_moving_object_positions = room_distribution.sample((num_samples, num_objects))
        initial_state = {
            'moving_object_positions': initial_moving_object_positions
        }
        return(initial_state)

    def transition_model_sample(self, current_state, current_time, next_time, parameters):
        minimum_drift = self.minimum_drift
        room_corners = tf.constant(self.room_corners, dtype=tf.float32)
        current_moving_object_positions = current_state['moving_object_positions']
        reference_time_interval = parameters['reference_time_interval']
        reference_drift = parameters['reference_drift']
        time_difference = tf.cast(next_time - current_time, dtype=tf.float32)
        calculated_drift = reference_drift*tf.sqrt(time_difference/reference_time_interval)
        drift = tf.maximum(calculated_drift, minimum_drift)
        drift_distribution = tfp.distributions.TruncatedNormal(
            loc = current_moving_object_positions,
            scale = drift,
            low = room_corners[0],
            high= room_corners[1]
        )
        next_moving_object_positions = drift_distribution.sample()
        next_state = {
            'moving_object_positions': next_moving_object_positions
        }
        return(next_state)

    def object_positions_no_fixed_dimensions(self, state, parameters):
        moving_object_positions = state['moving_object_positions']
        object_positions = moving_object_positions
        return object_positions

    def object_positions_with_fixed_dimensions(self, state, parameters):
        num_objects = self.num_objects
        num_fixed_object_dimensions = self.num_fixed_object_dimensions
        moving_object_positions = state['moving_object_positions']
        fixed_object_positions = parameters['fixed_object_positions']
        num_samples = tf.shape(moving_object_positions)[0]
        fixed_object_positions_reshaped = tf.reshape(fixed_object_positions, (num_objects, num_fixed_object_dimensions))
        fixed_object_positions_expanded = tf.expand_dims(fixed_object_positions_reshaped, axis = 0)
        fixed_object_positions_repeated = tf.tile(fixed_object_positions_expanded, (num_samples, 1, 1))
        object_positions = tf.concat((moving_object_positions, fixed_object_positions_repeated), axis = -1)
        return object_positions

    def measurement_value_distribution_function(self, state, parameters):
        anchor_positions = parameters['anchor_positions']
        object_positions = self.object_positions_function(state, parameters)
        relative_positions = tf.subtract(
            tf.expand_dims(object_positions, axis = 1),
            tf.expand_dims(tf.expand_dims(anchor_positions, 0), axis = 2))
        distances = tf.norm(relative_positions, axis = -1)
        measurement_value_means = self.measurement_value_mean_function(distances)
        measurement_value_sds = self.measurement_value_sd_function(distances)
        measurement_value_distribution = tfp.distributions.Normal(
            loc = measurement_value_means,
            scale = measurement_value_sds)
        return(measurement_value_distribution)

    def observation_model_sample_all_measurement_values(self, state, parameters):
        measurement_value_distribution = self.measurement_value_distribution_function(state, parameters)
        all_measurement_values = measurement_value_distribution.sample()
        observation_all_measurement_values = {
            self.measurement_value_name: all_measurement_values
        }
        return(observation_all_measurement_values)

    def observation_model_sample_all_successful(self, state, parameters):
        observation_all_measurement_values = self.observation_model_sample_all_measurement_values(state, parameters)
        observation = observation_all_measurement_values
        return(observation)

    def observation_model_sample_one_successful(self, state, parameters):
        observation_all_measurement_values = self.observation_model_sample_all_measurement_values(state, parameters)
        all_measurement_values = observation_all_measurement_values[self.measurement_value_name]
        num_samples = tf.shape(all_measurement_values)[0]
        num_elements = tf.size(all_measurement_values[0])
        logits = tf.zeros([num_samples, num_elements])
        choices = tf.transpose(tf.random.categorical(logits, 1))[0]
        range_vector = tf.cast(tf.range(num_samples), tf.int64)
        indices = tf.stack([range_vector, choices], axis = 1)
        ones_flat = tf.scatter_nd(indices, tf.ones([num_samples]), [num_samples, num_elements])
        trues_flat = tf.cast(ones_flat, dtype = tf.bool)
        trues = tf.reshape(trues_flat, tf.shape(all_measurement_values))
        all_nan = tf.fill(tf.shape(all_measurement_values), np.nan)
        chosen_values = tf.where(trues, all_measurement_values, all_nan)
        observation = {
            self.measurement_value_name: chosen_values
        }
        return(observation)

    def observation_model_sample_some_successful(self, state, parameters):
        ping_success_rate = parameters['ping_success_rate']
        observation_all_measurement_values = self.observation_model_sample_all_measurement_values(state, parameters)
        all_measurement_values = observation_all_measurement_values[self.measurement_value_name]
        ones = tfp.distributions.Bernoulli(probs = ping_success_rate).sample(tf.shape(all_measurement_values))
        trues = tf.cast(ones, dtype = tf.bool)
        all_nan = tf.fill(tf.shape(all_measurement_values), np.nan)
        chosen_values = tf.where(trues, all_measurement_values, all_nan)
        observation = {
            self.measurement_value_name: chosen_values
        }
        return(observation)

    def observation_model_pdf(self, state, observation, parameters):
        measurement_values = observation[self.measurement_value_name]
        measurement_value_distribution = self.measurement_value_distribution_function(state, parameters)
        log_pdfs = measurement_value_distribution.log_prob(measurement_values)
        log_pdfs_nans_removed = tf.where(tf.is_nan(log_pdfs), tf.zeros_like(log_pdfs), log_pdfs)
        log_pdf = tf.reduce_sum(log_pdfs_nans_removed, [-2, -1])
        return(log_pdf)

    def state_summary(self, state, log_weights, resample_indices, parameters):
        moving_object_positions = state['moving_object_positions']
        moving_object_positions_squared = tf.square(moving_object_positions)
        log_weights_renormalized = log_weights - tf.reduce_logsumexp(log_weights)
        weights_renormalized = tf.exp(log_weights_renormalized)
        weights_sum = tf.reduce_sum(weights_renormalized)
        moving_object_positions_mean = tf.tensordot(weights_renormalized, moving_object_positions, 1)
        moving_object_positions_squared_mean = tf.tensordot(weights_renormalized, moving_object_positions_squared, 1)
        moving_object_positions_var = moving_object_positions_squared_mean - tf.square(moving_object_positions_mean)
        moving_object_positions_sd = tf.sqrt(moving_object_positions_var)
        unique_resample_indices, _ = tf.unique(resample_indices)
        num_resample_indices = tf.size(unique_resample_indices)
        moving_object_positions_mean_expanded = tf.expand_dims(moving_object_positions_mean, 0)
        moving_object_positions_sd_expanded = tf.expand_dims(moving_object_positions_sd, 0)
        num_resample_indices_expanded = tf.expand_dims(num_resample_indices, 0)
        weights_sum_expanded = tf.expand_dims(weights_sum, 0)
        state_summary = {
            'moving_object_positions_mean': moving_object_positions_mean_expanded,
            'moving_object_positions_sd': moving_object_positions_sd_expanded,
            'num_resample_indices': num_resample_indices_expanded,
            'weights_sum': weights_sum_expanded
        }
        return state_summary

def parameter_structure_generator(num_anchors, num_objects, num_moving_object_dimensions, num_fixed_object_dimensions):
    num_dimensions = num_moving_object_dimensions + num_fixed_object_dimensions
    parameter_structure = {
        'fixed_object_positions': {
            'shape': [num_objects, num_fixed_object_dimensions],
            'type': 'float32'
        },
        'anchor_positions': {
            'shape': [num_anchors, num_dimensions],
            'type': 'float32'
        },
        'reference_time_interval': {
            'shape': [],
            'type': 'float32'
        },
        'reference_drift': {
            'shape': [],
            'type': 'float32'
        },
        'ping_success_rate': {
            'shape': [],
            'type': 'float32'
        }
    }
    return parameter_structure

def state_structure_generator(num_objects, num_moving_object_dimensions):
    state_structure = {
        'moving_object_positions': {
            'shape': [num_objects, num_moving_object_dimensions],
            'type': 'float32'
        }
    }
    return state_structure

def observation_structure_generator(num_anchors, num_objects, measurement_value_name):
    observation_structure = {
        measurement_value_name: {
            'shape': [num_anchors, num_objects],
            'type': 'float32'
        }
    }
    return observation_structure

def state_summary_structure_generator(num_objects, num_moving_object_dimensions):
    state_summary_structure = {
        'moving_object_positions_mean': {
            'shape': [num_objects, num_moving_object_dimensions],
            'type': 'float32'
        },
        'moving_object_positions_sd': {
            'shape': [num_objects, num_moving_object_dimensions],
            'type': 'float32'
        },
        'num_resample_indices': {
            'shape': [],
            'type': 'int32'
        },
        'weights_sum': {
            'shape': [],
            'type': 'float32'
        }
    }
    return state_summary_structure
