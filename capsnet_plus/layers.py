import tensorflow as tf
import numpy as np


epsilon = 1e-9


class ConvCapsLayer(object):
    def __init__(self, batch_size, num_outputs=32, vec_length=8):
        self.num_outputs = num_outputs
        self.vec_length = vec_length
        self.batch_size = batch_size

    def __call__(self, input, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        capsules = tf.contrib.layers.conv2d(input, num_outputs=self.num_outputs * self.vec_length,
                                           kernel_size=kernel_size, stride=stride, padding='VALID',
                                           activation_fn=tf.nn.relu)
        capsules = tf.reshape(capsules, (self.batch_size, -1, self.vec_length, 1))
        capsules_ = squash(capsules)
        return capsules, capsules_


class FCCapsLayer(object):
    def __init__(self, batch_size, num_outputs=32, vec_length=8):
        self.num_outputs = num_outputs
        self.vec_length = vec_length
        self.batch_size = batch_size

    def __call__(self, input):
        self.input = tf.reshape(input, shape=(self.batch_size, -1, 1, input.shape[-2].value, 1))
        with tf.variable_scope('routing'):
            b = tf.constant(np.zeros(shape=(self.batch_size, input.shape[1].value, self.num_outputs, 1, 1)), dtype=tf.float32)
            capsules = routing(self.input, b, self.batch_size, self.num_outputs)
            capsules_ = tf.squeeze(capsules, axis=1)
        return capsules, capsules_


def routing(input, b, batch_size, num_output):
    w = tf.get_variable('Weight', shape=(1, input.shape[1].value, 16 * num_output, 8, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.01))
    biases = tf.get_variable('Bias', shape=[1, 1, num_output, 16, 1], dtype=tf.float32)
    input = tf.tile(input, multiples=[1, 1, 16 * num_output, 1, 1])
    # assert input.get_shape() == [batch_size, 1152, 16 * num_output, 8, 1]
    u_hat = tf.reduce_sum(w * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=(-1, input.shape[1].value, num_output, 16, 1))
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
    num_iterations = 3
    for r_iter in range(num_iterations):
        with tf.variable_scope('iter_' + str(r_iter)):
            c_i = tf.nn.softmax(b, axis=2)
            if r_iter == num_iterations - 1:
                s_i = tf.multiply(c_i, u_hat)
                s_i = tf.reduce_sum(s_i, axis=1, keepdims=True) + biases
                v_j = squash(s_i)
            else:

                s_i = tf.multiply(c_i, u_hat_stopped)
                s_i = tf.reduce_sum(s_i, axis=1, keepdims=True) + biases
                v_j = squash(s_i)

                v_j_tiled = tf.tile(v_j, multiples=[1, input.shape[1].value, 1, 1, 1])
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_j_tiled, axis=3, keepdims=True)
                b += u_produce_v
    return v_j


def squash(vector):
    vec_squared_norm = tf.reduce_sum(tf.square(vector), axis=2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    squashed = scalar_factor * vector
    return squashed
