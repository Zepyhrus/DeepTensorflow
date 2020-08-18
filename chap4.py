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
      # 存储变量W, 注意只有最后一次的结果被保存到.ckpt文件中
      saver.save(sess, './tmp/summary/test.ckpt')

def exp4_2_3_restore():
  W = tf.Variable(0.0, name='weights')
  double = tf.multiply(2.0, W)

  # 创建Saver
  saver = tf.train.Saver()
  with tf.Session() as sess:
    # 恢复变量W的值
    saver.restore(sess, './tmp/summary/test.ckpt')
    print('restored W = %s' % W.eval())

    for i in range(4):
      sess.run(tf.assign_add(W, 1.0))
      print('W=%s, double=%s' % (W.eval(), double.eval()))


def conv_relu(input, kernel_shape, bias_shape):
  # 创建或获取名叫weights/bias的变量
  weights = tf.get_variable('weights', kernel_shape, 
    initializer=tf.random_normal_initializer())
  
  bias = tf.get_variable('biases', bias_shape,
    initializer=tf.random_uniform_initializer())
  
  conv = tf.nn.conv2d(input, weights,
    strides=[1, 1, 1, 1], padding='SAME')
  
  return tf.nn.relu(conv + biases)


def my_image_filter(input_images):
  with tf.variable_scope('conv1', reuse=True):
    # 创建conv1/weights和conv1/biases变量
    relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
  
  with tf.variable_scope('conv2', reuse=True):
    # 创建conv2/weights和conv2/biases变量
    return conv_relu(relu1, [5, 5, 32, 32], [32])


def exp4_2_4():
  with tf.variable_scope('foo', initializer=tf.constant_initializer(0.4)):
    v = tf.get_variable('v', [1])
    # 注意这里必须在v作为一个Variable被添加到Graph中后方可通过global_variables_initializer初始化
    # 使用外层定义的初始化方法
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      assert v.eval() == 0.4
      print(v.name)

    W = tf.get_variable('W', [1], initializer=tf.constant_initializer(0.3))
    # 显式地初始化覆盖了
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      
      assert W.eval() == 0.3
    with tf.variable_scope('bar'):
      v = tf.get_variable('v', [1])
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 继承外层作用域的初始化方法
        assert v.eval() == 0.4

    with tf.variable_scope('baz', initializer=tf.constant_initializer(0.2)):
      v = tf.get_variable('v', [1])
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        assert v.eval() == 0.2
        print(v.name)



if __name__ == "__main__":
  exp4_2_4()


