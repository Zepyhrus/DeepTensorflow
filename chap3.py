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

def example3_4_1():
  x = tf.placeholder(tf.float32)
  W = tf.Variable(1.0)
  b = tf.Variable(1.0)
  y = W * x + b

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    fetch = y.eval(feed_dict={x: 3.0})

    print(fetch)

def example3_5_3():
  def minimize(
    self, loss, global_step=None, val_list=None,
    gate_gradients=GATE_OP, aggregation_method=None,
    colocate_gradients_with_ops=False, name=None,
    grad_loss=None
  ):
    # 计算梯度，得到组合后的梯度值与模型参数列表——grads_and_vars
    # 即<梯度，参数>键值对列表
    grads_and_vars = self.compute_gradients(
      loss, var_list=val_list, gate_gardients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops,
      grad_loss=grad_loss
    )
    # 从grads_and_vars中取出非零梯度值对应的模型参数列表——vars_with_grad
    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    # 如果没有非零梯度值，则说明模型计算过程中出现了问题
    if not vars_with_grad:
      raise ValueError("..." % ([str(v) for _, v in grads_and_vars], loss))

    # 使用非零梯度值更新对应的模型参数
    return self.apply_gradients(
      grads_and_vars, global_step=global_step, name=name
    )

  X = tf.placeholder(...)
  Y_ = tf.placeholder(...)
  w = tf.Variable(...)
  b = tf.Variable(...)
  Y = tf.matmul(X, w) + b

  # 使用交叉熵作为损失函数
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)
  )

  # 优化器
  optimizer = tf.train.GradientDecentOptimizer(learning_rate=0.01)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in xrange(max_train_steps):
      sess.run(train_op, feed_dict={...})

    # 训练日志
    if step % log_steps == 0:
      final_loss, weight, bias = sess.run(
        [loss, w, b], feed_dict={...}
      )
      print('Step: ..., loss = ..., w = ..., b = ..., step' \
        % (step, final_loss, weight, bias))

if __name__ == "__main__":
  example3_4_1()


