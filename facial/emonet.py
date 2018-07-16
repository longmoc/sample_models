import tensorflow as tf


class EmoNet(object):
    def __init__(self, image_size, num_classes):
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, image_size, image_size, 1), name='input')
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None, num_classes), name='one_hot')
        self.num_classes = num_classes
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self._get_model()

    def _get_model(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        with tf.variable_scope('FirstPart'):
            forward = tf.contrib.layers.conv2d(self.input, num_outputs=8, kernel_size=3, stride=1, padding='SAME',
                                               weights_regularizer=regularizer, biases_initializer=None,
                                               normalizer_fn=tf.contrib.layers.batch_norm, activation_fn=tf.nn.relu)
            # forward = tf.contrib.layers.batch_norm(forward, activation_fn=tf.nn.relu)

            forward = tf.contrib.layers.conv2d(forward, num_outputs=8, kernel_size=3, stride=1, padding='SAME',
                                               weights_regularizer=regularizer, biases_initializer=None,
                                               normalizer_fn=tf.contrib.layers.batch_norm, activation_fn=tf.nn.relu)
            # forward = tf.contrib.layers.batch_norm(forward, activation_fn=tf.nn.relu)

        out_size = 16
        for i in range(4):
            forward = self.residual(forward, out_size, i)
            out_size = out_size * 2

        with tf.variable_scope('LastPart'):
            forward = tf.contrib.layers.conv2d(forward, num_outputs=self.num_classes, kernel_size=3, padding='SAME')
            forward = tf.contrib.layers.avg_pool2d(forward, kernel_size=1)
            forward = tf.reduce_mean(forward, axis=[1, 2])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=forward, labels=self.labels)

        pred = tf.nn.softmax(forward, name='prediction')
        pred_cls = tf.argmax(pred, dimension=1, name='prediction_cls')
        self.cost = tf.reduce_mean(cross_entropy, name='loss')
        true_cls = tf.argmax(self.labels, dimension=1)
        correct_prediction = tf.equal(pred_cls, true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        self._summary()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = self.optimizer.minimize(self.cost, global_step=self.global_step)

    def residual(self, input_tensor, out_size, block_num):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        with tf.variable_scope('ResidualBlock_{}'.format(block_num)):
            res = tf.contrib.layers.conv2d(input_tensor, num_outputs=out_size, kernel_size=1, stride=2, padding='SAME',
                                           normalizer_fn=tf.contrib.layers.batch_norm, biases_initializer=None)

            depth_wise = tf.contrib.layers.separable_conv2d(input_tensor, depth_multiplier=1, num_outputs=out_size,
                                                            kernel_size=3, stride=1, padding='SAME',
                                                            weights_regularizer=regularizer,
                                                            normalizer_fn=tf.contrib.layers.batch_norm,
                                                            activation_fn=tf.nn.relu,
                                                            biases_initializer=None)

            depth_wise = tf.contrib.layers.separable_conv2d(depth_wise, depth_multiplier=1, num_outputs=out_size,
                                                            kernel_size=3, stride=1, padding='SAME',
                                                            weights_regularizer=regularizer,
                                                            normalizer_fn=tf.contrib.layers.batch_norm,
                                                            activation_fn=None,
                                                            biases_initializer=None)
            depth_wise = tf.contrib.layers.max_pool2d(depth_wise, kernel_size=3, stride=2, padding='SAME')
            return depth_wise + res

    def _summary(self):
        loss_sum = tf.summary.scalar('train/loss', self.cost)
        acc_sum = tf.summary.scalar('train/acc', self.accuracy)
        train_summary = [loss_sum, acc_sum]
        self.train_summary = tf.summary.merge(train_summary)
