import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

parameter_structure = {
    'room_dimensions':{
        'shape': [2],
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
        'shape': [4, 2],
        'dtype': tf.float32
    },
    'reference_distance': {
        'shape': [],
        'dtype': tf.float32
    },
    'reference_rssi': {
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
        'shape': [3, 2],
        'dtype': tf.float32
    }
}

observation_structure = {
    'rssis': {
        'shape': [4, 3],
        'dtype': tf.float32
    }
}

def parameter_model_sample():
    parameters = {
        'room_dimensions': tf.constant([10.0, 20.0], dtype=tf.float32),
        'num_objects': tf.constant(3, dtype=tf.int32),
        'reference_time_interval': tf.constant(1.0, dtype=tf.float32),
        'reference_drift': tf.constant(2.0, dtype=tf.float32),
        'num_anchors': tf.constant(4, dtype = tf.int32),
        'anchor_positions': tf.constant(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 20.0],
                [10.0, 20.0]
            ],
            dtype = tf.float32),
        'reference_distance': tf.constant(1.0, dtype=tf.float32),
        'reference_rssi': tf.constant(-60.0, dtype=tf.float32),
        'rssi_std_dev': tf.constant(5.0, dtype=tf.float32)
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
    reference_rssi = parameters['reference_rssi']
    rssi_std_dev = parameters['rssi_std_dev']
    relative_positions = tf.subtract(
        tf.expand_dims(positions, axis = 1),
        tf.expand_dims(tf.expand_dims(anchor_positions, 0), axis = 2))
    distances = tf.norm(relative_positions, axis = -1)
    log10_distances = tf.log(distances)/tf.log(10.0)
    log10_reference_distance = tf.log(reference_distance)/tf.log(10.0)
    mean_rssis = reference_rssi + (-20.0)*(log10_distances - log10_reference_distance)
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
    reference_rssi = parameters['reference_rssi']
    rssi_std_dev = parameters['rssi_std_dev']
    relative_positions = tf.subtract(
        tf.expand_dims(positions, axis = 1),
        tf.expand_dims(tf.expand_dims(anchor_positions, 0), axis = 2))
    distances = tf.norm(relative_positions, axis = -1)
    log10_distances = tf.log(distances)/tf.log(10.0)
    log10_reference_distance = tf.log(reference_distance)/tf.log(10.0)
    mean_rssis = reference_rssi + (-20.0)*(log10_distances - log10_reference_distance)
    rssi_distribution = tfp.distributions.Normal(
        loc = mean_rssis,
        scale = rssi_std_dev)
    log_pdfs = rssi_distribution.log_prob(rssis)
    log_pdf = tf.reduce_sum(log_pdfs, [-2, -1])
    return(log_pdf)
