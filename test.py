import tensorflow as tf
import os

from capsNet import CapsNet
from utils import load_data

image_size = 28
logdir = './train'

num_epochs = 2000
batch_size = 128


def main():
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data('mnist', batch_size, is_training=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    model = CapsNet()
    sv = tf.train.Supervisor(graph=model.graph, logdir=logdir, save_model_secs=0)
    with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        for epoch in range(num_epochs):
            print('Ep %d/%d' % (epoch, num_epochs))
            if sv.should_stop():
                print('Should stop.')
                break
            for step in range(num_tr_batch):
                global_step = epoch * num_tr_batch + step
                # model.X, model.Y, _, _ = data.train.next_batch(batch_size)
                if global_step % 10 == 0:
                    _, loss, acc, summary, margin_loss = sess.run(
                        [model.train_op, model.total_loss, model.accuracy, model.train_summary, model.margin_loss])
                    print('Step %d: loss %f \t acc %f \t margin loss %f' % (global_step, loss, acc, margin_loss))
                    # l, ml, mr, v_length, digit_cap, primary_caps, pre_vec, conv1= sess.run([model.Y, model.max_l,
                    #                                                                  model.max_r, model.v_length,
                    #                                                                  model.digit_caps, model.primary_caps,
                    #                                                                  model.pre_vector, model.conv1])
                    # print(l[0])
                    # # print(v_length[0])
                    # # print(ml[0])
                    # # print(mr[0])
                    # # print(digit_cap[0])
                    # # print(primary_caps[0])
                    # print('conv1')
                    # print(conv1[0][0])

                    sv.summary_writer.add_summary(summary, global_step)
                else:
                    sess.run(model.train_op)


if __name__ == '__main__':
    main()
