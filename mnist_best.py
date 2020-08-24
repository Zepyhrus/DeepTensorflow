from tqdm import tqdm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data/mnist', 'Directory for storing mnist data')
flags.DEFINE_float('learning_rate', 0.5, 'Learning rate')
flags.DEFINE_integer('max_step', 10000, 'max training step')
flags.DEFINE_integer('interval', 100, 'message/save interval')
FLAGS = flags.FLAGS



def main():
  # 创建MNIST数据集实例
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # 创建模型
  # 输入模块
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
  
  with tf.name_scope('input_reshape'):
    # 将输入图像x转换成四阶张量
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    # 添加获取手写体图像的汇总操作，设置最大生成10张图像
    tf.summary.image('input', image_shaped_input, 10)

  # softmax网络层
  with tf.name_scope('softmax_layer'):
    with tf.name_scope('weights'):
      W = tf.Variable(tf.zeros([784, 10]))
      # 添加模型权重值的汇总操作
      tf.summary.histogram('weights', W)
    with tf.name_scope('biases'):
      b = tf.Variable(tf.zeros([10]))
      # 添加模型偏置值的汇总操作
      tf.summary.histogram('biases', b)
    with tf.name_scope('Wx_plus_b'):
      y = tf.matmul(x, W) + b
  

  # 使用交叉熵作为损失值
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
      tf.summary.scalar('cross_entropy', cross_entropy)

  # # 如果不使用合并的交叉熵及softmax则
  # with tf.name_scope('cross_entropy'):
  #   _ot = tf.softmax(y)
  #   ce = tf.reduce_mean(
  #     -tf.reduce_sum(y_ * tf.log(_ot), reduction_indices=[1])
  #   )

  # 优化器
  with tf.name_scope('train'):
    # 创建梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

    # 定义单步训练操作
    train_op = optimizer.minimize(cross_entropy)

  # 准确率
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_predicion = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_predicion, tf.float32))
      tf.summary.scalar('accuracy', accuracy)

  # 聚集汇总操作
  merged = tf.summary.merge_all()

  # 创建Saver
  saver = tf.train.Saver()
  sess = tf.InteractiveSession()
  train_writer = tf.summary.FileWriter('tmp/summary/mnist/train', sess.graph)
  test_writer = tf.summary.FileWriter('tmp/summary/mnist/test')

  # 全局初始化
  tf.global_variables_initializer().run()
  # # or 从ckpt中恢复
  # saver.restore(sess, 'mnist.ckpt')

  def feed_dict(train):
    """填充训练数据或测试数据的方法"""
    if train:
      xs, ys = mnist.train.next_batch(100, fake_data=False)
    else:
      xs, ys = mnist.test.images, mnist.test.labels
    
    return {x: xs, y_: ys}

  # 最大训练步数设为 max_step
  for i in range(FLAGS.max_step):
    if i % 10 == 0: # 写汇总数据和测试集的准确率
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
    else:
      if i % 100 == 99: # 写运行时的事件数据
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run(
          [merged, train_op],
          feed_dict=feed_dict(True),
          options=run_options,
          run_metadata=run_metadata
        )
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:
        summary, _ = sess.run([merged, train_op], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()

if __name__ == "__main__":
  main()


