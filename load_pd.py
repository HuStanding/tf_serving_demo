# !/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Copyright (c) 2020 ke.com, Inc. All Rights Reserved
# *
# * File:load_pd.py
# * Description:
# * Author:huzhu
# * Email:huzhuo002@ke.com
# * CreatedTime:2021-01-04 23:30
# **************************************************************************
import tensorflow as tf

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ["serve"], "./pb_model")
    graph = tf.get_default_graph()

    x = sess.graph.get_tensor_by_name("input_x:0")
    y = sess.graph.get_tensor_by_name("input_y:0")

    a = sess.graph.get_tensor_by_name("output_a:0")
    b = sess.graph.get_tensor_by_name("output_b:0")

    result_add = sess.run(a, feed_dict={x: [[1, 2]], y: [[3, 4]]})
    result_sub = sess.run(b, feed_dict={x: [[1, 2]], y: [[3, 4]]})

print(result_add)
print(result_sub)
