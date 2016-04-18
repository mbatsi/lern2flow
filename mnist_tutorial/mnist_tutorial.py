# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 07:43:24 2016

@author: atreo
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

###########################
### MATHEMATICAL MODEL ####
# Create placeholder for image
x = tf.placeholder(tf.float32, [None, 784])

# Initialize the model's parameters as variables - these will be learned!
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Convert to probabilities with softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define a cost function with cross entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Define optimization algorithm
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

###########################


###########################
#### TRAIN THE MODEL ####
# Initialize all variables
init = tf.initialize_all_variables()

# Launch model in a session
sess = tf.Session()
sess.run(init)

# Stochastic training
# Instead of using all data at once to train, let's use 1000 batches of 100 datapoints
for i in range(1000):
    # Get 100 random data points from the training set
    # x, y_ are placeholders -> feed them with data
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
############################
    
###########################
#### EVALUATE THE TRAINED MODEL ####    

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))