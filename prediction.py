#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper


def predict(test_data, checkpoint_file, vocab_filepath, batch_size):
    inpH = InputHelper()
    x1_test,x2_test,y_test = inpH.getTestDataSet(test_data, vocab_filepath, 30)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement= False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/distance").outputs[0]
            accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
            sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]
            batches = inpH.batch_iter(list(zip(x1_test,x2_test,y_test)), 2*batch_size, 1, shuffle=False)
            all_predictions = []
            all_d=[]
            for db in batches:
                x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
                batch_predictions, batch_acc, batch_sim = sess.run([predictions,accuracy,sim], {input_x1: x1_dev_b, input_x2: x2_dev_b, input_y:y_dev_b, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                all_d = np.concatenate([all_d, batch_sim])
            return all_predictions, all_d
