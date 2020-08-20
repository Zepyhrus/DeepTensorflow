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
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  y_ = tf.placeholder(tf.float32, [None, 10])

  # 使用交叉熵作为损失值
  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  )

  # 如果不使用合并的交叉熵及softmax则
  _ot = tf.softmax(y)
  ce = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(_ot), reduction_indices=[1])
  )

  # 创建梯度下降优化器
  optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

  # 定义单步训练操作
  train_op = optimizer.minimize(cross_entropy)


if __name__ == "__main__":
  main()


