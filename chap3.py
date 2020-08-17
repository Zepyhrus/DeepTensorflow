import numpy as np

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
    indices=[[0, 0], [0, 2], [1, 1]],
    values=[1, 1, 1],
    dense_shape=[2, 3]
  )

  reduce_x = [
    tf.sparse_reduce_sum(sp),                         # => 3
    tf.sparse_reduce_sum(sp, axis=1),                 # => [2, 1]
    tf.sparse_reduce_sum(sp, axis=1, keep_dims=True), # => [[2], [1]]
    tf.sparse_reduce_sum(sp, axis=[0, 1])             # => 3
  ]

  with tf.Session() as sess:
    print(sess.run(reduce_x))


def example3_3_1():
  # help(tf.reshape)

  with tf.name_scope('AddExample'):
    a = tf.Variable(1.0, name='a')
    b = tf.Variable(2.0, name='b')

    c = tf.add(a, b, name='add')

    print(c)


def variable_op_v2(shape, dtype, name='Variable', container='', shared_name=''):
  """Source code of variable operation in TF"""
  return gen_state_ops._variable_v2(
    shape=shape,
    dtype=dtype,
    name=name,
    container=container,
    shared_name=shared_name
  )

def example3_3_3():
  x = tf.placeholder(tf.float32, shape=(2, 2), name='x')
  y = tf.matmul(x, x)

  with tf.Session() as sess:
    print(sess.run(y, feed_dict={x: np.random.rand(2, 2)}))

  x = tf.sparse_placeholder(tf.float32)
  y = tf.sparse_reduce_sum(x)

  with tf.Session() as sess:
    indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    shape = np.array([7, 9, 2], dtype=np.int64)

    print(
      sess.run(y, feed_dict={
        x: tf.SparseTensorValue(indices, values, shape)
      })
    )

    print(
      sess.run(y, feed_dict={x: (indices, values, shape)})
    )

    sp = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    sp_value = sp.eval()

    print(
      sess.run(y, feed_dict={x: sp_value})
    )




if __name__ == "__main__":
  example3_3_3()


