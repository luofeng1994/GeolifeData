import tensorflow as tf
import numpy as np

init = tf.global_variables_initializer()
print('\n\n3-D')
a = tf.constant(np.arange(1, 10), shape=[1, 10],dtype=tf.int32)
e = tf.Variable(initial_value=0.0)
b = a[0][0]
c = a[0][1]
d = a[0][2]
e = b+c+d
f = tf.nn.relu(a)
with tf.Session() as sess:
    sess.run(init)

    print('a=', a.eval())
    # print('b=', b.eval())
    # print('c=', c.eval())
    # print('d=', d.eval())
    # print('e=', e.eval())
    print('f=', e.eval())