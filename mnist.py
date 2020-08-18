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
  


if __name__ == "__main__":
  main()


