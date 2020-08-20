import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as tf_session
'''
# Dummy BaseSession code
class BaseSession(SessionInterface):
  def __init__(self, target='', graph=None, config=None):
    # 
    self._session = None
    opts = tf_session.TF_NewSessionOptions(self._target, config=config)
    # 
    try:
      with errors.raise_exception_on_not_ok_status() as status:
        # 
        self._session = tf_session.TF_NewSession(opts, status)
    finally:
      tf_session.TF_DeleteSessionOptions(opts)
'''



if __name__ == "__main__":
  g1 = tf.Graph()

  with g1.as_default():
    # 
    a = tf.Variable(0, name='a')

    assert a.graph is g1

  with tf.Graph().as_default() as g2:
    # 
    b = tf.Variable(0, name='b')

    assert b.graph is g2