#Parampreet Singh

#Matrix Multiplication using TensorFlow Libraries

import tensorflow as tf

#Matrix1 of size 1*2
a=tf.constant([[3.,3.]])

#Matrix2 of size 2*1
b=tf.constant([[2.],[2.]])

#Matrix3 of size 1*1
c=tf.matmul(a,b)
sess1=tf.Session()
print(sess1.run(c))
sess1.close()

#Matrix4 of size 2*2
d=tf.matmul(b,a)
sess2=tf.Session()
print(sess2.run(d))
sess2.close()



