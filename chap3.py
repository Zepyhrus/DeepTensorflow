import tensorflow as tf


def example3_2_1():
  a = tf.constant(1.0)  
  b = tf.constant(2.0)
  c = tf.add(a, b)

  print([a, b, c])

  with tf.Session() as sess:
    print(c.eval())
    print(sess.run([a, b, c]))

  
  a = tf.constant([1, 1])
  b = tf.constant([2, 2])
  c = tf.add(a, b)

  with tf.Session() as sess:
    print('-' * 32)
    print(f'a[0]={a[0].eval()}, a[1]={a[1].eval()}')  # this will add additional operation on a.consumers
    print(f'c.name={c.name}')
    print(f'c.value={c.eval()}')
    print(f'c.shape={c.shape}')
    print(f'a.consumers={a.consumers()}')
    print(f'b.consumers={b.consumers()}')
    print(f'[c.op]:\n{c.op}')


def example3_2_2():
  sp = tf.SparseTensor(
    indices=[[0, 2], [1, 3]],
    values=[1, 2],
    dense_shape=[3, 4]
  )

  with tf.Session() as sess:
    print(sp.eval())



if __name__ == "__main__":
  example3_2_2()


