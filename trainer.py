"""trainer.py"""
from tensorflow import flags
import tensorflow as tf

# 定义TF集群参数
flags.DEFINE_integer('task_index', 0, 'Worker task index')
flags.DEFINE_string('ps_hosts', '127.0.0.1:22',
  'Comma-separated list of hostname:port pairs')
flags.DEFINE_string('worker_hosts', '123.206.2.25:22',
  'Comma-separated list of hostname:port pairs')
flags.DEFINE_string('job_name', None, 'job name: worker or PS')
FLAGS = flags.FLAGS


def main(unused_argv):
  # 解析集群参数ps_hosts和worker_hosts
  PS_spec = FLAGS.ps_hosts.split(',')
  worker_spec = FLAGS.worker_hosts.split(',')

  # 定义TF集群
  cluster = tf.tarin.ClusterSpec({
    'PS': PS_spec,
    'worker': worker_spec
  })

  server = tf.train.server(
    cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index
  )

  if FLAGS.job_name == 'PS':
    server.join()

  is_chief = (FLAGS.task_index == 0)



if __name__ == "__main__":
  main(None)
