'''chapter 4: writer.py'''
import tensorflow as tf

if __name__ == "__main__":
  # 创建向TFRecords文件写数据的writer
  writer = tf.python_io.TFRecordWriter('stat.tfrecord')

  # 构造输入样例
  for i in range(1, 3):
    # 创建example.proto中定义的样例
    example = tf.train.Example(
      features = tf.train.Features(
        feature = {
          'id': tf.train.Feature(int64_list = tf.train.Int64List(value=[i])),
          'age': tf.train.Feature(int64_list = tf.train.Int64List(value=[i*24])),
          'income': tf.train.Feature(float_list = tf.train.FloatList(value=[i*2048.0])),
          'outgo': tf.train.Feature(float_list = tf.train.FloatList(value=[i*1024.0]))
        }
      )
    )

    # 将样例序列化为字符串后，写入stat.tfrecord文件
    writer.write(example.SerializeToString())
  # 关闭输出流
  writer.close()


