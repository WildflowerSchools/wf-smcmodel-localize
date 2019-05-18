import smcmodel
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def localization_model(
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
    reference_distance = 1.0,
    reference_mean_rssi = -60.0,
    mean_rssi_slope = -20.0,
    rssi_std_dev = 5.0,
    ping_success_rate = 1.0):
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
    parameter_structure = {
        'num_objects': {
            'shape': [],
            'type': 'int32'
        },
        'num_anchors': {
            'shape': [],
            'type': 'int32'
        },
        'num_moving_object_dimensions': {
            'shape': [],
            'type': 'int32'
        },
        'num_fixed_object_dimensions': {
            'shape': [],
            'type': 'int32'
        },
        'fixed_object_positions': {
            'shape': [num_objects, num_fixed_object_dimensions],
            'type': 'float32'
        },
        'room_corners': {
            'shape': [2, num_moving_object_dimensions],
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
        'reference_distance': {
            'shape': [],
            'type': 'float32'
        },
        'reference_mean_rssi': {
            'shape': [],
            'type': 'float32'
        },
        'rssi_std_dev': {
            'shape': [],
            'type': 'float32'
        },
        'ping_success_rate': {
            'shape': [],
            'type': 'float32'
        }
    }

    state_structure = {
        'moving_object_positions': {
            'shape': [num_objects, num_moving_object_dimensions],
            'type': 'float32'
        }
    }
    observation_structure = observation_structure_generator(num_anchors, num_objects)
    # observation_structure = {
    #     'rssis': {
    #         'shape': [num_anchors, num_objects],
    #         'type': 'float32'
    #     }
    # }

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
        }
    }

    def parameter_model_sample():
        parameters = {
            'num_objects': tf.constant(num_objects, dtype=tf.int32),
            'num_anchors': tf.constant(num_anchors, dtype = tf.int32),
            'num_moving_object_dimensions': tf.constant(num_moving_object_dimensions, dtype = tf.int32),
            'num_fixed_object_dimensions': tf.constant(num_fixed_object_dimensions, dtype = tf.int32),
            'fixed_object_positions': tf.constant(fixed_object_positions, dtype=tf.float32),
            'room_corners': tf.constant(room_corners, dtype=tf.float32),
            'anchor_positions': tf.constant(anchor_positions, dtype = tf.float32),
            'reference_time_interval': tf.constant(reference_time_interval, dtype=tf.float32),
            'reference_drift': tf.constant(reference_drift, dtype=tf.float32),
            'reference_distance': tf.constant(reference_distance, dtype=tf.float32),
            'reference_mean_rssi': tf.constant(reference_mean_rssi, dtype=tf.float32),
            'mean_rssi_slope': tf.constant(mean_rssi_slope, dtype=tf.float32),
            'rssi_std_dev': tf.constant(rssi_std_dev, dtype=tf.float32),
            'ping_success_rate': tf.constant(ping_success_rate, dtype=tf.float32)
        }
        return parameters

    def initial_model_sample(num_samples, parameters):
        room_corners = parameters['room_corners']
        num_objects = parameters['num_objects']
        room_distribution = tfp.distributions.Uniform(
            low = room_corners[0],
            high= room_corners[1]
        )
        initial_moving_object_positions = room_distribution.sample((num_samples, num_objects))
        initial_state = {
            'moving_object_positions': initial_moving_object_positions
        }
        return(initial_state)

    def transition_model_sample(current_state, current_time, next_time, parameters):
        current_moving_object_positions = current_state['moving_object_positions']
        reference_time_interval = parameters['reference_time_interval']
        reference_drift = parameters['reference_drift']
        room_corners = parameters['room_corners']
        time_difference = tf.cast(next_time - current_time, dtype=tf.float32)
        drift = reference_drift*tf.sqrt(time_difference/reference_time_interval)
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

    def object_positions_no_fixed_dimensions(state, parameters):
        moving_object_positions = state['moving_object_positions']
        object_positions = moving_object_positions
        return object_positions

    def object_positions_with_fixed_dimensions(state, parameters):
        moving_object_positions = state['moving_object_positions']
        fixed_object_positions = parameters['fixed_object_positions']
        num_samples = tf.shape(moving_object_positions)[0]
        fixed_object_positions_reshaped = tf.reshape(fixed_object_positions, (num_objects, num_fixed_object_dimensions))
        fixed_object_positions_expanded = tf.expand_dims(fixed_object_positions_reshaped, axis = 0)
        fixed_object_positions_repeated = tf.tile(fixed_object_positions_expanded, (num_samples, 1, 1))
        object_positions = tf.concat((moving_object_positions, fixed_object_positions_repeated), axis = -1)
        return object_positions

    if num_fixed_object_dimensions == 0:
        object_positions_function = object_positions_no_fixed_dimensions
    else:
        object_positions_function = object_positions_with_fixed_dimensions

    def rssi_distribution_function(state, parameters):
        # moving_object_positions = state['moving_object_positions']
        anchor_positions = parameters['anchor_positions']
        reference_distance = parameters['reference_distance']
        reference_mean_rssi = parameters['reference_mean_rssi']
        mean_rssi_slope = parameters['mean_rssi_slope']
        rssi_std_dev = parameters['rssi_std_dev']
        object_positions = object_positions_function(state, parameters)
        relative_positions = tf.subtract(
            tf.expand_dims(object_positions, axis = 1),
            tf.expand_dims(tf.expand_dims(anchor_positions, 0), axis = 2))
        distances = tf.norm(relative_positions, axis = -1)
        log10_distances = tf.log(distances)/tf.log(10.0)
        log10_reference_distance = tf.log(reference_distance)/tf.log(10.0)
        mean_rssis = reference_mean_rssi + mean_rssi_slope*(log10_distances - log10_reference_distance)
        rssi_distribution = tfp.distributions.Normal(
            loc = mean_rssis,
            scale = rssi_std_dev)
        return(rssi_distribution)

    def observation_model_sample_all_rssis(state, parameters):
        rssi_distribution = rssi_distribution_function(state, parameters)
        all_rssis = rssi_distribution.sample()
        observation_all_rssis = {
            'rssis': all_rssis
        }
        return(observation_all_rssis)

    def observation_model_sample_all_successful(state, parameters):
        observation_all_rssis = observation_model_sample_all_rssis(state, parameters)
        observation = observation_all_rssis
        return(observation)

    def observation_model_sample_one_successful(state, parameters):
        observation_all_rssis = observation_model_sample_all_rssis(state, parameters)
        all_rssis = observation_all_rssis['rssis']
        num_samples = tf.shape(all_rssis)[0]
        num_elements = tf.size(all_rssis[0])
        logits = tf.zeros([num_samples, num_elements])
        choices = tf.transpose(tf.random.categorical(logits, 1))[0]
        range_vector = tf.cast(tf.range(num_samples), tf.int64)
        indices = tf.stack([range_vector, choices], axis = 1)
        ones_flat = tf.scatter_nd(indices, tf.ones([num_samples]), [num_samples, num_elements])
        trues_flat = tf.cast(ones_flat, dtype = tf.bool)
        trues = tf.reshape(trues_flat, tf.shape(all_rssis))
        all_nan = tf.fill(tf.shape(all_rssis), np.nan)
        chosen_values = tf.where(trues, all_rssis, all_nan)
        observation = {
            'rssis': chosen_values
        }
        return(observation)

    def observation_model_sample_some_successful(state, parameters):
        ping_success_rate = parameters['ping_success_rate']
        observation_all_rssis = observation_model_sample_all_rssis(state, parameters)
        all_rssis = observation_all_rssis['rssis']
        ones = tfp.distributions.Bernoulli(probs = ping_success_rate).sample(tf.shape(all_rssis))
        trues = tf.cast(ones, dtype = tf.bool)
        all_nan = tf.fill(tf.shape(all_rssis), np.nan)
        chosen_values = tf.where(trues, all_rssis, all_nan)
        observation = {
            'rssis': chosen_values
        }
        return(observation)

    if ping_success_rate == 1.0:
        observation_model_sample = observation_model_sample_all_successful
    elif ping_success_rate == 0.0:
        observation_model_sample = observation_model_sample_one_successful
    elif ping_success_rate > 0.0 and ping_success_rate < 1.0:
        observation_model_sample = observation_model_sample_some_successful
    else:
        raise ValueError('Ping success rate out of range')

    def observation_model_pdf(state, observation, parameters):
        rssis = observation['rssis']
        rssi_distribution = rssi_distribution_function(state, parameters)
        log_pdfs = rssi_distribution.log_prob(rssis)
        log_pdfs_nans_removed = tf.where(tf.is_nan(log_pdfs), tf.zeros_like(log_pdfs), log_pdfs)
        log_pdf = tf.reduce_sum(log_pdfs_nans_removed, [-2, -1])
        return(log_pdf)

    def state_summary(state, log_weights, resample_indices, parameters):
        moving_object_positions = state['moving_object_positions']
        moving_object_positions_squared = tf.square(moving_object_positions)
        weights = tf.exp(log_weights)
        weights_sum = tf.reduce_sum(weights)
        moving_object_positions_mean = tf.tensordot(weights, moving_object_positions, 1)/weights_sum
        moving_object_positions_squared_mean = tf.tensordot(weights, moving_object_positions_squared, 1)/weights_sum
        moving_object_positions_var = moving_object_positions_squared_mean - tf.square(moving_object_positions_mean)
        moving_object_positions_sd = tf.sqrt(moving_object_positions_var)
        unique_resample_indices, _ = tf.unique(resample_indices)
        num_resample_indices = tf.size(unique_resample_indices)
        moving_object_positions_mean_expanded = tf.expand_dims(moving_object_positions_mean, 0)
        moving_object_positions_sd_expanded = tf.expand_dims(moving_object_positions_sd, 0)
        num_resample_indices_expanded = tf.expand_dims(num_resample_indices, 0)
        state_summary = {
            'moving_object_positions_mean': moving_object_positions_mean_expanded,
            'moving_object_positions_sd': moving_object_positions_sd_expanded,
            'num_resample_indices': num_resample_indices_expanded
        }
        return state_summary

    model = smcmodel.SMCModelGeneralTensorflow(
        parameter_structure,
        state_structure,
        observation_structure,
        state_summary_structure,
        parameter_model_sample,
        initial_model_sample,
        transition_model_sample,
        observation_model_sample,
        observation_model_pdf,
        state_summary
    )
    return model

def observation_structure_generator(num_anchors, num_objects):
    observation_structure = {
        'rssis': {
            'shape': [num_anchors, num_objects],
            'type': 'float32'
        }
    }
    return observation_structure
