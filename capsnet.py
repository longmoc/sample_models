import tensorflow as tf
from layers import ConvCapsLayer, FCCapsLayer
from utils import get_batch_data

m_positive = 0.9
m_negative = 0.1
lambda_ = 0.5
regularization_scale = 0.0005 * 784

from dataset import get_data_batch, load_train


class CapsNet(object):
    def __init__(self, train_path, image_size, classes, batch_size):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.num_classes = len(classes)
            images, labels, _, _ = load_train(train_path, image_size, classes)
            self.X, self.labels = get_data_batch(images, labels, batch_size, num_threads=8)
            self.Y = tf.one_hot(self.labels, depth=self.num_classes, axis=1, dtype=tf.float32)
            self.batch_size = batch_size

            # self.X, self.labels = get_batch_data('mnist', batch_size, 4)
            # self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)

            self.build_arch()
            self.loss()
            self.model_summary()

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

    def build_arch(self):
        # Encode
        with tf.variable_scope('Conv1_layer'):
            # Input: 28*28*1 (with mnist)
            # Conv: (size 9*9*1) * (256 kernels) / (stride 1)
            # Output: 20*20*256 tensor
            # Num of params = 20992
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256, kernel_size=9, stride=1, padding='VALID')
        with tf.variable_scope('PrimaryCaps'):
            # Input: 20*20*256 tensor
            # 32 primary capsules
            # Conv: (size 9*9) * (256 kernels)
            # Output: 6*6*8*32 tensor
            primary = ConvCapsLayer(self.batch_size, num_outputs=32, vec_length=8)
            self.pre_vector, self.primary_caps = primary(conv1, kernel_size=9, stride=2)

        with tf.variable_scope('DigitCaps'):
            # Input: 6*6*8*32

            digit = FCCapsLayer(self.batch_size, num_outputs=self.num_classes, vec_length=16)
            self.pre_dig, self.digit_caps = digit(self.primary_caps)

        # Masking
        with tf.variable_scope('Masking'):
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=2, keepdims=True) + 1e-9)
            self.softmax_v = tf.nn.softmax(self.v_length, axis=1, name='y_pred')
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(self.batch_size,), name='y_pred_cls')
            self.masked_v = tf.multiply(tf.squeeze(self.digit_caps), tf.reshape(self.Y, (-1, self.num_classes, 1)))

        # Decoder
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(self.batch_size, -1))
            fc_1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            fc_2 = tf.contrib.layers.fully_connected(fc_1, num_outputs=1024)
            self.decoded = tf.contrib.layers.fully_connected(fc_2, num_outputs=784, activation_fn=tf.sigmoid)

    def loss(self):
        max_l = tf.square(tf.maximum(0., m_positive - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - m_negative))

        max_l = tf.reshape(max_l, shape=(self.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(self.batch_size, -1))

        Tk = self.Y
        Lk = Tk * max_l + lambda_ * (1 - Tk) * max_r
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(Lk, axis=1))

        orgin = tf.reshape(self.X, shape=(self.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        self.total_loss = self.margin_loss + regularization_scale * self.reconstruction_err

    def model_summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        reconstruction_img = tf.reshape(self.decoded, shape=(self.batch_size, 28, 28, 1))
        train_summary.append(tf.summary.image('reconstruction_img', reconstruction_img))
        self.train_summary = tf.summary.merge(train_summary)

        self.correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(self.correct_prediction, tf.float32)) / self.batch_size
