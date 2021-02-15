# !/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Copyright (c) 2020 ke.com, Inc. All Rights Reserved
# *
# * File:save_to_cpkt.py
# * Description:
# * Author:huzhu
# * Email:huzhuo002@ke.com
# * CreatedTime:2021-01-04 22:49
# **************************************************************************

'''
checkpoint 文本文件，记录了模型文件的路径信息列表
model.ckpt.data-00000-of-00001 网络参数值
model.ckpt.index 文件保存了当前参数名和索引
model.ckpt.meta 保存模型的网络结构
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
result_del = sess.run(b, feed_dict={x: [[1, 2]], y: [[3, 4]]})

print(result_add)
print(result_del)

tf.train.Saver().save(sess, './ckpt_model/model.ckpt')
