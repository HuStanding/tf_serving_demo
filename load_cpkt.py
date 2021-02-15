# !/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Copyright (c) 2020 ke.com, Inc. All Rights Reserved
# *
# * File:load_cpkt.py
# * Description:
# * Author:huzhu
# * Email:huzhuo002@ke.com
# * CreatedTime:2021-01-04 23:16
# **************************************************************************
import tensorflow as tf

ckpt = tf.train.get_checkpoint_state('./ckpt_model/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)

    x = sess.graph.get_tensor_by_name("input_x:0")
    y = sess.graph.get_tensor_by_name("input_y:0")

    a = sess.graph.get_tensor_by_name("output_a:0")
    b = sess.graph.get_tensor_by_name("output_b:0")

    result_add = sess.run(a, feed_dict={x: [[1, 2]], y: [[3, 4]]})
    result_sub = sess.run(b, feed_dict={x: [[1, 2]], y: [[3, 4]]})

print(result_add)
print(result_sub)
