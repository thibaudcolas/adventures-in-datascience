#!/usr/bin/env python
# See https://www.tensorflow.org/get_started/get_started

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# tf.train API

# Model parameters
# Defining variables that will evolve over the training of the model.
# Rank 1, simple vectors of scalar values.
a = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = a * x + b
y = tf.placeholder(tf.float32)

# Loss
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# Initialise all variables in the TensorFlow program.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Training loop
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

# Evaluate training accuracy
curr_a, curr_b, curr_loss = sess.run([a, b, loss], {x:x_train, y:y_train})
# a: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
print("a: %s b: %s loss: %s"%(curr_a, curr_b, curr_loss))
