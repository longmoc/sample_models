import os
import tensorflow as tf
from emonet import EmoNet
from emo_datasets import DataManager, split_data
from preprocessor import preprocess_input
from dataset import DataSet

logdir = 'emo_train'
num_epochs = 100
validation_split = 0.2
batch_size = 256
input_shape = (64, 64, 1)


def load_data(data_name, path):
    data_loader = DataManager(data_name, image_size=input_shape[:2], dataset_path=path)
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    image_size = faces.shape[1]
    train_data, val_data = split_data(faces, emotions, validation_split)
    return train_data, val_data, image_size, num_classes


def validate(val_set=None, num_val_batch=None):
    if not val_set:
        train_data, val_data, image_size, num_classes = load_data('fer2013', '../data/fer2013/fer2013.csv')
        val_faces, val_emotions = val_data
        val_set = DataSet(val_faces, val_emotions, None, None)
        num_val_batch = int(val_emotions.shape[0] / batch_size)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        checkpoint = tf.train.latest_checkpoint(logdir)
        saver = tf.train.import_meta_graph(checkpoint + '.meta')
        saver.restore(sess, checkpoint)
        total_acc = 0
        for step in range(num_val_batch):
            example_batch, label_batch = val_set.next_batch_v2(batch_size)
            acc = sess.run('accuracy:0', feed_dict={'input:0': example_batch, 'one_hot:0': label_batch})
            total_acc += acc
        print('Accuracy: ', total_acc / num_val_batch)


def main():
    train_data, val_data, image_size, num_classes = load_data('fer2013', '../data/fer2013/fer2013.csv')
    train_faces, train_emotions = train_data
    train_set = DataSet(train_faces, train_emotions, None, None)
    num_tr_batch = int(train_emotions.shape[0] / batch_size) + 1

    val_faces, val_emotions = val_data
    val_set = DataSet(val_faces, val_emotions, None, None)
    num_val_batch = int(val_emotions.shape[0] / batch_size) + 1

    model = EmoNet(image_size, num_classes)
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # graph = tf.get_default_graph()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        writer = tf.summary.FileWriter(logdir)
        writer.add_graph(sess.graph)
        sess.run(init_op)
        global_step = 0
        if os.path.isdir(logdir):
            try:
                saver.restore(sess, tf.train.latest_checkpoint(logdir))
                global_step = sess.run('global_step:0')
            except ValueError:
                pass
        for epoch in range(num_epochs):
            print('Ep %d/%d' % (epoch + 1, num_epochs))
            for step in range(num_tr_batch):
                global_step += 1
                example_batch, label_batch = train_set.next_batch_v2(batch_size)
                if global_step % 10 == 0:
                    _, loss, acc, summary = sess.run([model.train_op, model.cost, model.accuracy, model.train_summary],
                                                     feed_dict={model.input: example_batch, model.labels: label_batch})
                    print('Step %d: loss %f \t acc %f' % (global_step, loss, acc))
                    writer.add_summary(summary, global_step)
                else:
                    sess.run(model.train_op, feed_dict={model.input: example_batch, model.labels: label_batch})
            if (epoch + 1) % 2 == 0:
                saver.save(sess, logdir + '/model_%02d' % global_step)

            if (epoch + 1) % 2 == 0:
                # Validation
                total_acc = 0
                for step in range(num_val_batch):
                    example_batch, label_batch = val_set.next_batch_v2(batch_size)
                    acc = sess.run('accuracy:0', feed_dict={'input:0': example_batch, 'one_hot:0': label_batch})
                    total_acc += acc
                print('Accuracy: ', total_acc / num_val_batch)


if __name__ == '__main__':
    main()
