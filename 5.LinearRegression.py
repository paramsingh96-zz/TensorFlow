# Parampreet Singh

# Program to implement LinearRegression Learning Algorithm using TensorFlow in Python

# Importing Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Defining our training variables
trX = np.linspace(-1,1,101)
# Adding some noise to y
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

# plotting our training x & y
plt.plot(trX, trY)
plt.show()



# A placeholder is a value that we'll input when we ask TensorFlow to run a computation

X = tf.placeholder("float")
Y = tf.placeholder("float")

# Defining our LinearRegression Model
def model(X, W, b):
	return tf.add( tf.mul(X, W), b)

W = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

y_model = model(X, W, b)

# Defining our cost model
cost = tf.square(Y - y_model)

# Defining our optimising algorithm for LinearRegression
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Initialising all variables and running session for execution
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100):
	for(x, y) in zip(trX, trY):
		sess.run(train_op, feed_dict={X: x, Y: y})
	

# Printing the optimised values of W and b
print(sess.run(W))
print(sess.run(b))

# The value of W must be around 2
# Run this program for many times to have great insight in LinearRegression. :)
