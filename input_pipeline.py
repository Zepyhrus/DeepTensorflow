'''Persudo code'''
import tensorflow as tf

'''
def get_my_example(filename_queue):
  reader = tf.SomeReader()
  _, value = reader.read(filename_queue)
  features = tf.decode_some(value)
  # 对样例进行预处理
  processed_example = some_processing(features)
  return processed_example


def input_pipeline(filenames, batch_size, num_epochs=None):
  # 当num_epochs==None时，表示文件名队列总是可用的，一直循环入队
  filename_queue = tf.train.string_input_producer(
    filenames, num_epochs=num_epochs, shuffle=True
  )
  example = get_my_example(filename_queue)
  # min_after_dequeue表示从样例队列中出队的样例个数，
  # 值越大表示大乱的顺序效果越好，同时意味着消耗更多的内存
  min_after_dequeue = 10000
  # capacity表示批数据队列的容量，推荐设置：
  # min_after_dequeue + (num_threads + a small safety margin) * batch_size
  capacity = min_after_dequeue + 3 * batch_size
  # 创建批样例example_batch
  example_batch = tf.train.shuffle_batch(
    [example], batch_size=batch_size, capacity=capacity,
    min_after_dequeue=min_after_dequeue
  )

  return example_batch


if __name__ == "__main__":
  x_batch = input_pipeline(['stat.tfrecord'], batch_size=20)

  # 省略创建模型的步骤
  sess = tf.Session()
  init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
  )
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
'''

