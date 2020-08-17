'''chapter 4: reader.py'''
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == "__main__":
  # 创建文件名队列filename_queue
  filename_queue = tf.train.string_input_producer(['stat.tfrecord'], num_epochs=2)  # num_epochs for debugging
  
  # 创建读取TFRecords文件的reader
  reader = tf.TFRecordReader()

  # 取出stat.tfrecord文件中的一条序列化的样例serialized_example
  _, serialized_example = reader.read(filename_queue)
  
  # 将一条序列化的样例转换为起爆含的所有特征张量
  features = tf.parse_single_example(
    serialized_example,
    features = {
      'id': tf.FixedLenFeature([], tf.int64),
      'age': tf.FixedLenFeature([], tf.int64),
      'income': tf.FixedLenFeature([], tf.float32),
      'outgo': tf.FixedLenFeature([], tf.float32)
    }
  )

  print(features)

  # 初始化协调器需要通过local_variable_initializer进行
  init_op = tf.group([
    tf.global_variables_initializer(), tf.local_variables_initializer()
  ])
  sess = tf.Session()

  sess.run(init_op)

  # 创建协调器
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  # 打印程序的后台线程信息
  print('Threads: %s' % threads)

  try:
    for i in range(10):
      if not coord.should_stop():
        example = sess.run(features)
        print(example)
  except tf.errors.OutOfRangeError:
    print('Catch OutOfRangeError')
  finally:
    # 请求停止所有后台线程
    coord.request_stop()
    print('Finish reading')
  
  # 等待所有后台线程安全退出
  coord.join(threads)
  sess.close()


  