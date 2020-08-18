import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def exp4_2_2():
  W = tf.Variable(0.0, name='W')
  double = tf.multiply(W, 2.0)

  with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(4):
      print(sess.run(double))

      # NOTE: we must run assign to actually operate on W
      sess.run(tf.assign_add(W, 1.0))


def exp4_2_3():
  W = tf.Variable(0.0, name='W')
  double = tf.multiply(2.0, W)

  # 创建Saver
  saver = tf.train.Saver({'weights': W})
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(4):
      sess.run(tf.assign_add(W, 1.0))
      # 存储变量W
      saver.save(sess, './tmp/summary/test.ckpt')

if __name__ == "__main__":
  exp4_2_3()


