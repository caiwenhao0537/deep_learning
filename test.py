import tensorflow as tf

with tf.variable_scope("foo"):
    v1=tf.get_variable("v",[1])
    print(v1.name)

with tf.variable_scope("",reuse=True):
    v2=tf.get_variable("foo/v",[1])
    print(v2==v1)