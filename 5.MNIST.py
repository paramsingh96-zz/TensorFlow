# Parampreet Singh

# Simple tutorial for understanding MNIST DataSets

# Importing different libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

# MNIST are the data sets of hand written images ranging from 0 to 9
# The data is split into three parts:  55,000 data points of training data
# 10,000 data points of test data
# 5,000 of validation data 
# Each image is 28 pixel by 28 pixel
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Within each, we can access images, labels and num_examples 
print(mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)

# (55000, 784) (55000, 10) 
print(mnist.train.images.shape, mnist.train.labels.shape)

# The ranges of values in each images is from 0-1
print(np.min(mnist.train.images), np.max(mnist.train.images))

# We can visualize any one of the images by reshaping it to a 28*28 image
plt.imshow(np.reshape(mnist.train.images[1000, :], (28, 28)), cmap='rainbow')
plt.show()

# try out with different data points and colors, we will get great insights. :)




