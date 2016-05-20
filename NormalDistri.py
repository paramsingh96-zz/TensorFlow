#Parampreet Singh

#Program to find The Normal Distribution Curve using TensorFlow Libraries

import tensorflow as tf
import matplotlib.pyplot as plt

n=64
#Creating an array 
x=tf.linspace(-3.0,3.0,n)
sess=tf.InteractiveSession()


#Defining the parameters of a Standard Normal Random Variable
sigma=1.0
mean=0.0

#Defining the probability distribution of Normal Random Variable
z=(tf.exp(tf.neg(tf.pow(x - mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))) * (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

#New Operations are added as Default Graph
assert z.graph is tf.get_default_graph()

#Executing graph and plotting it
plt.plot(z.eval())

#Showing the plotted graph 
plt.show()

