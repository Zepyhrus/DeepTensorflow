from tqdm import tqdm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data/mnist', 'Directory for storing mnist data')
flags.DEFINE_float('learning_rate', 0.5, 'Learning rate')
FLAGS = flags.FLAGS



def main():
  # 创建MNIST数据集实例
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # 创建模型
  with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    with tf.device('/gpu:0'):
      y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 10])

  # 使用交叉熵作为损失值
  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  )

  # # 如果不使用合并的交叉熵及softmax则
  # _ot = tf.softmax(y)
  # ce = tf.reduce_mean(
  #   -tf.reduce_sum(y_ * tf.log(_ot), reduction_indices=[1])
  # )

  # 创建梯度下降优化器
  optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

  # 定义单步训练操作
  train_op = optimizer.minimize(cross_entropy)

  # 创建Saver
  saver = tf.train.Saver()
  sess = tf.InteractiveSession()
  # 全局初始化
  tf.global_variables_initializer().run()
  # # or 从ckpt中恢复
  # saver.restore(sess, 'mnist.ckpt')

  # 最大训练步数设为10000
  for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
    # 每100步保存一次模型参数
    if i % 100 == 0:
      saver.save(sess, 'tmp/mnist/mnist.ckpt')
      correct_predicion = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_predicion, tf.float32))
      print('acc=%s' % sess.run(accuracy,
        feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == "__main__":
  main()


