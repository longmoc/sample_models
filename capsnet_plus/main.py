import os
import cv2
import tensorflow as tf
import numpy as np
import dataset
from capsnet_plus.capsnet import CapsNet

image_size = 45
validation_size = 0.1
train_path = 'data/alpha_pixels/'
classes = os.listdir(train_path)
num_classes = len(classes)
logdir = './capsnet_plus/new_train'

num_epochs = 12
batch_size = 16


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    image = image[:, :, 0:1].reshape((image_size, image_size, 1))
    return image


def evaluate(valid_data, batch_size):
    model = CapsNet(train_path, image_size, classes, batch_size)
    num_valid_batch = int(valid_data.num_examples / batch_size) + 1
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    supervisor = tf.train.Supervisor(graph=model.graph, logdir=logdir, save_model_secs=0)
    with supervisor.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(logdir))
        tf.logging.info('Model restored!')
        test_acc = 0
        for step in range(num_valid_batch):
            x, y, _, _ = valid_data.next_batch(batch_size)
            acc = sess.run(model.accuracy, {model.X: x, model.labels: y})
            test_acc += acc
        test_acc = test_acc / num_valid_batch
        print(test_acc)


def eval_single_image(image_path):
    image = read_image(image_path)
    model = CapsNet(train_path, image_size, classes, 1)
    sv = tf.train.Supervisor(graph=model.graph, logdir=logdir, save_model_secs=0)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sv.saver.restore(sess, tf.train.latest_checkpoint(logdir))
        tf.logging.info('Model restored!')
        y_pred, y_pred_cls = sess.run((model.softmax_v, model.argmax_idx), {model.X: [image]})
        print(y_pred[0])
        print(y_pred_cls[0], y_pred[0][y_pred_cls[0]][0][0])


def train():
    data = dataset.read_train_sets(train_path, image_size, classes, validation_size=validation_size)
    num_tr_batch = int(data.train.num_examples / batch_size) + 1
    # trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data('mnist', batch_size, is_training=True)
    model = CapsNet(train_path, image_size, classes, batch_size)
    sv = tf.train.Supervisor(graph=model.graph, logdir=logdir, save_model_secs=0)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        global_step = 0
        for epoch in range(num_epochs):
            print('Ep %d/%d' % (epoch, num_epochs - 1))
            if sv.should_stop():
                print('Should stop.')
                break
            for step in range(num_tr_batch):
                global_step = epoch * num_tr_batch + step
                if global_step % 10 == 0:
                    _, loss, acc, summary, margin_loss = sess.run(
                        [model.train_op, model.total_loss, model.accuracy, model.train_summary, model.margin_loss])
                    print('Step %d: loss %f \t acc %f \t margin loss %f' % (global_step, loss, acc, margin_loss))
                    sv.summary_writer.add_summary(summary, global_step)
                else:
                    sess.run(model.train_op)

            if (epoch + 1) % 2 == 0:
                sv.saver.save(sess, logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        evaluate(data.valid, batch_size=10)


def test():
    data = dataset.read_train_sets(train_path, image_size, classes, validation_size=0.9)
    evaluate(data.valid, batch_size=batch_size)


if __name__ == '__main__':
    # train()
    # test()
    eval_single_image('test/d_t2.jpg')
