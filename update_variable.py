import tensorflow as tf

def demo():
  W = tf.Variable(initial_value=tf.random_normal(shape=(1, 4), mean=100, stddev=0.35), name='W')

  w_replica = tf.Variable(W.initial_value(), name='w_replica')
  w_twice = tf.Variable(W.initial_value() * 2.0, name='w_twice')

if __name__ == "__main__":
  weights = tf.Variable(tf.random_normal(shape=(1, 4), stddev=0.35), name='weights')
  biases = tf.Variable(tf.zeros([4]), name='biases')

  # 
  with tf.Session() as sess:
    sess.run(tf.variables_initializer([weights, biases]))
    print(sess.run(weights), sess.run(biases))





