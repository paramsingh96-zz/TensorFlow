'''
    Author= 'Parampreet Singh'
    Project= 'https://github.com/paramsingh96/TensorFlow-Tutorials'
    
    Program to find the accuracy of prediction of digits by using SoftmaxRegression on MNIST data using TensorFlow Libraries
'''

# Use Python3 for simplicity

# Importing the TensorFlow
import tensorflow as tf


# Downloading the MNIST data(handwritten images data sets)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# A placeholder is a value that we'll input when we ask TensorFlow to run a computation. 
# x is a 2-D Tensor of floating-point numbers with shape [None, 784]
# None means that a dimension can be of any length
x = tf.placeholder(tf.float32, [None, 784])

# A variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations.
# W and b are tensors full of zeros.
# W has shape [784,10] because we want to multiply 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence for different classes of b.
# b has shape [10] so we can add it to the output. 
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Our SoftmaxRegression Model

# First, x is multiplied by W. Then it is flipped and added to b
# y is the predicted probability distribution
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Training

# let y_ be the true probability distribution i.e. one_hot vector
y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy tells how inefficient our predictions are for describing the truth.
# tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1]
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Since TensorFlow knows the entire graph of our computations, it is automatically using ~BackpropagationAlgorithm~ to efficiently determine how our variables affect the cost we ask to minimize.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# This is minimizing cross_entropy with learning rate 0.5 using gradient descent algorithm
# Gradient descent is a simple procedure, where TensorFlow simply shifts each variable a little bit in the direction that reduces the cost.
# TensorFlow, behind the scenes, adds new operation to our graph which implement ~backpropagation~ and ~GradientDescent~. 
# Then it gives back a single operation which do a step of gradient descent training by tweaking our variables to reduce the cost.

# Initializing all variables
init = tf.initialize_all_variables()

# Launching the model in the Session
sess = tf.Session()
sess.run(init)

# Training
# Running the training step 1000 times
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step,	feed_dict={x: batch_xs, y_: batch_ys})
# Each step of loop, we get a "batch" of one hundred random data points from our training set.
# We run train_step feeding in the batches data to replace the placeholders.

# Evaluating our model
# correct_prediction gives us a list of booleans
# tf.argmax gives us the index of the highest entry in a tensor along some axis
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# accuracy of our data
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# percent accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# This should be around 92%
# ~~ Run several times this program, you'll get different accuracies.
# And then, take the insights. :)
