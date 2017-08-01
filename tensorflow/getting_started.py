#!/usr/bin/env python
# See https://www.tensorflow.org/get_started/get_started

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
# print(node1, node2)

sess = tf.Session()
# [3.0, 4.0]
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
# print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

add_and_triple = adder_node * 3.

print(sess.run(add_and_triple, {a: 3, b:4.5}))

print(sess.run(add_and_triple, {a: [1,3], b: [2, 4]}))

print("Variables")

a = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = a * x + b

# Initialise all variables in the TensorFlow program.
init = tf.global_variables_initializer()
sess.run(init)

# [ 0.          0.30000001  0.60000002  0.90000004]
print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print("loss value", sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

fixa = tf.assign(a, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixa, fixb])
print("loss value", sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
