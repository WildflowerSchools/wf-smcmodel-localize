import smcmodel
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def localization_model(
    num_objects = 3,
    num_anchors = 4,
    num_dimensions = 2,
    room_dimensions = [10.0, 20.0],
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
    rssi_std_dev = 5.0):
    parameter_structure = {
        'room_dimensions':{
            'shape': [num_dimensions],
            'dtype': tf.float32
        },
        'num_objects':{
            'shape': [],
            'dtype': tf.int32
        },
        'reference_time_interval': {
            'shape': [],
            'dtype': tf.float32
        },
        'reference_drift': {
            'shape': [],
            'dtype': tf.float32
        },
        'anchor_positions': {
            'shape': [num_anchors, num_dimensions],
            'dtype': tf.float32
        },
        'reference_distance': {
            'shape': [],
            'dtype': tf.float32
        },
        'reference_mean_rssi': {
            'shape': [],
            'dtype': tf.float32
        },
        'rssi_std_dev': {
            'shape': [],
            'dtype': tf.float32
        }
    }
    state_structure = {
        'positions': {
            'shape': [num_objects, num_dimensions],
            'dtype': tf.float32
        }
    }
    observation_structure = {
        'rssis': {
            'shape': [num_anchors, num_objects],
            'dtype': tf.float32
        }
    }
    def parameter_model_sample():
        parameters = {
            'room_dimensions': tf.constant(room_dimensions, dtype=tf.float32),
            'num_objects': tf.constant(num_objects, dtype=tf.int32),
            'reference_time_interval': tf.constant(reference_time_interval, dtype=tf.float32),
            'reference_drift': tf.constant(reference_drift, dtype=tf.float32),
            'num_anchors': tf.constant(num_anchors, dtype = tf.int32),
            'anchor_positions': tf.constant(anchor_positions, dtype = tf.float32),
            'reference_distance': tf.constant(reference_distance, dtype=tf.float32),
            'reference_mean_rssi': tf.constant(reference_mean_rssi, dtype=tf.float32),
            'mean_rssi_slope': tf.constant(mean_rssi_slope, dtype=tf.float32),
            'rssi_std_dev': tf.constant(rssi_std_dev, dtype=tf.float32)
        }
        return parameters
    def initial_model_sample(num_samples, parameters):
        room_dimensions = parameters['room_dimensions']
        num_objects = parameters['num_objects']
        room_distribution = tfp.distributions.Uniform(
            low = [0.0, 0.0],
            high= room_dimensions
        )
        initial_positions = room_distribution.sample((num_samples, num_objects))
        initial_state = {
            'positions': initial_positions
        }
        return(initial_state)
    def transition_model_sample(previous_state, previous_time, current_time, parameters):
        previous_positions = previous_state['positions']
        reference_time_interval = parameters['reference_time_interval']
        reference_drift = parameters['reference_drift']
        room_dimensions = parameters['room_dimensions']
        drift = reference_drift*((current_time - previous_time)/reference_time_interval)
        drift_distribution = tfp.distributions.TruncatedNormal(
            loc = previous_positions,
            scale = drift,
            low = [0.0, 0.0],
            high = room_dimensions
        )
        current_positions = drift_distribution.sample()
        current_state = {
            'positions': current_positions
        }
        return(current_state)

    def observation_model_sample(state, parameters):
        positions = state['positions']
        anchor_positions = parameters['anchor_positions']
        reference_distance = parameters['reference_distance']
        reference_mean_rssi = parameters['reference_mean_rssi']
        mean_rssi_slope = parameters['mean_rssi_slope']
        rssi_std_dev = parameters['rssi_std_dev']
        relative_positions = tf.subtract(
            tf.expand_dims(positions, axis = 1),
            tf.expand_dims(tf.expand_dims(anchor_positions, 0), axis = 2))
        distances = tf.norm(relative_positions, axis = -1)
        log10_distances = tf.log(distances)/tf.log(10.0)
        log10_reference_distance = tf.log(reference_distance)/tf.log(10.0)
        mean_rssis = reference_mean_rssi + mean_rssi_slope*(log10_distances - log10_reference_distance)
        rssi_distribution = tfp.distributions.Normal(
            loc = mean_rssis,
            scale = rssi_std_dev)
        rssis = rssi_distribution.sample()
        observation = {
            'rssis': rssis
        }
        return(observation)
    def observation_model_pdf(state, observation, parameters):
        positions = state['positions']
        rssis = observation['rssis']
        anchor_positions = parameters['anchor_positions']
        reference_distance = parameters['reference_distance']
        reference_mean_rssi = parameters['reference_mean_rssi']
        mean_rssi_slope = parameters['mean_rssi_slope']
        rssi_std_dev = parameters['rssi_std_dev']
        relative_positions = tf.subtract(
            tf.expand_dims(positions, axis = 1),
            tf.expand_dims(tf.expand_dims(anchor_positions, 0), axis = 2))
        distances = tf.norm(relative_positions, axis = -1)
        log10_distances = tf.log(distances)/tf.log(10.0)
        log10_reference_distance = tf.log(reference_distance)/tf.log(10.0)
        mean_rssis = reference_mean_rssi + mean_rssi_slope*(log10_distances - log10_reference_distance)
        rssi_distribution = tfp.distributions.Normal(
            loc = mean_rssis,
            scale = rssi_std_dev)
        log_pdfs = rssi_distribution.log_prob(rssis)
        log_pdf = tf.reduce_sum(log_pdfs, [-2, -1])
        return(log_pdf)
    model = smcmodel.SMCModelGeneralTensorflow(
        parameter_structure,
        state_structure,
        observation_structure,
        parameter_model_sample,
        initial_model_sample,
        transition_model_sample,
        observation_model_sample,
        observation_model_pdf
    )
    return model
