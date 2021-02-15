# !/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Copyright (c) 2020 ke.com, Inc. All Rights Reserved
# *
# * File:save_to_pd.py
# * Description:
# * Author:huzhu
# * Email:huzhuo002@ke.com
# * CreatedTime:2021-01-04 23:28
# **************************************************************************
'''
saved_model.pb 保存图形结构
variables 保存训练所习得的权重。
'''

import tensorflow as tf

x = tf.placeholder(tf.float32, [1, 2], name='input_x')
y = tf.placeholder(tf.float32, [1, 2], name='input_y')
z = tf.Variable([[1.0, 1.0]], name='var_z')
a = x + y
tf.identity(a, name="output_a")
b = x - y
tf.identity(b, name="output_b")

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

result_add = sess.run(a, feed_dict={x: [[1, 2]], y: [[3, 4]]})
result_sub = sess.run(b, feed_dict={x: [[1, 2]], y: [[3, 4]]})

print(result_add)
print(result_sub)

tf.saved_model.simple_save(sess,
                           "./pb_model/",
                           inputs={
                               "input_x": x,
                               "input_y": y
                           },
                           outputs={
                               "result_add": a,
                               "result_sub": b
                           })
