'''
    Author= 'Parampreet Singh'
    Project= 'https://github.com/paramsingh96/TensorFlow-Tutorials'
    HelloWorld program using TensorFlow Library
'''

import tensorflow as tf

hello=tf.constant('Hello World')

#Creating Sessions
sess=tf.Session()

print(sess.run(hello))

#Closing Session sess
sess.close()
