from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
from dataset import model_data
from core import model

# config device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5

# config dataset params (3 datasets with different scales)
params_data = {
    'mode': 'train',
    'batch': 3,
    'root_dir': '../../dataset'
}

# create dataset
with tf.device('/cpu:0'):
    my_dataset = model_data(params_data)
    # DEBUG
    trainset = my_dataset.get_trainset()

# config train params
params_model = {
    'batch': 3,
    'l2_weight': 0.0002,
    'init_lr': 1e-3, # original paper:
    'margin': 0.01, # default static margin
    'data_format': 'NCHW', # optimal for cudnn
    'save_path': '../data/ckpts/model.ckpt',
    'tsboard_logs': '../data/tsboard_logs/',
    'restore_model': '../data/ckpts/model.ckpt-xxx'
}

# define epochs
epochs = 5
num_samples = 7500 # approximately
steps_per_epochs = num_samples*epochs
global_step = 0
save_ckpt_interval = 1
summary_write_interval = 50
print_screen_interval = 20

# placeholder
batch = params_model['batch']
feed_triplets = tf.placeholder(tf.float32, [3*batch, 64, 64, 3])
# display current feed
feed_anchors = feed_triplets[0:batch, :, :, :]
feed_pos = feed_triplets[batch:2*batch, :, :, :]
feed_neg = feed_triplets[2*batch:3*batch, :, :, :]
sum_anchor = tf.summary.image('anchors', feed_anchors)
sum_pos = tf.summary.image('pos', feed_pos)
sum_neg = tf.summary.image('neg', feed_neg)
# DEBUG
dump_op = feed_triplets

# build network, on GPU by default
# my_model = model.model(params_model)
# loss, step = my_model.train(feed_triplets)
init_op = tf.global_variables_initializer()
sum_all = tf.summary.merge_all()

# define saver
# saver = tf.train.Saver()

# run session
with tf.Session(config=config_gpu) as sess:
    sum_writer = tf.summary.FileWriter(params_model['tsboard_logs'], sess.graph)
    sess.run(init_op)

    # restore all variables
    # saver.restore(sess, params_model['restore_model'])
    # print('restored variables from {}'.format(params_model['restore_model']))
    print("All weights initialized.")

    for i in range(3):
        next_batch = my_dataset.next_batch()
        print('step: {0}, feed shape: {1}'.format(i, next_batch.shape))
        feed_dict_v = {feed_triplets: next_batch}
        dump_op_, sum_all_ = sess.run([dump_op, sum_all], feed_dict=feed_dict_v)
        sum_writer.add_summary(sum_all_, i)
    print("Test complete.")

    # print("Start training for {0} epochs, {1} global steps.".format(epochs, num_samples*epochs))
    # for ep in range(epochs):
    #     print("Epoch {} ...".format(ep))
    #     for local_step in range(steps_per_epochs):
    #         # get next batch
    #         next_batch = my_dataset.next_batch()
    #         feed_dict_v = {feed_triplets: next_batch}
    #         # execute backprop
    #         loss_, step_, sum_all_ = sess.run([loss, step, sum_all], feed_dict=feed_dict_v)
    #
    #         # save summary
    #         if global_step % summary_write_interval == 0 and global_step !=0:
    #             sum_writer.add_summary(sum_all_, global_step)
    #         # print to screen
    #         if global_step % print_screen_interval == 0:
    #             print("Global step {0} loss: {1}".format(global_step, loss_))
    #         # save .ckpt
    #         if global_step % save_ckpt_interval == 0 and global_step != 0:
    #             saver.save(sess=sess,
    #                        save_path=params_model['save_path'],
    #                        global_step=global_step,
    #                        write_meta_graph=False)
    #             print('Saved checkpoint.')
    #         global_step += 1
    # print("Finished training.")



