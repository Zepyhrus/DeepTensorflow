import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True

import tensorflow as tf
import numpy as np
import os
from os.path import join, split

import cv2

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = '/home/ubuntu/Workspace/DeepTensorflow/minimalsample'
NAME_TO_VISUALIZE_VARIABLE = 'mnistembedding'
TO_EMBED_COUNT = 5000


def create_sprite_image(images):
  """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
  if isinstance(images, list):
    images = np.array(images)

  c, h, w = images.shape
  cols = min(int(np.ceil(np.sqrt(c))), 50)
  rows = (c - 1) // cols + 1
  # cols = int(np.ceil(np.sqrt(c)))
  # rows = cols
  
  spriteimage = np.ones((h * rows, w * cols))
  
  for i in range(rows):
    for j in range(cols):
      this_filter = i * cols + j
      if this_filter < c:
        this_img = images[this_filter]
        spriteimage[i * h:(i + 1) * h,
          j * w:(j + 1) * w] = this_img
    
  return spriteimage

def vector_to_matrix_mnist(mnist_digits):
  """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
  return np.reshape(mnist_digits,(-1, 28, 28))

def invert_grayscale(mnist_digits):
  """ Makes black white, and white black """
  return 1 - mnist_digits


if __name__ == "__main__":
  path_for_mnist_sprites = join(LOG_DIR, 'mnistdigits.png')
  path_for_mnist_metadata = join(LOG_DIR, 'metadata.tsv')

  mnist = input_data.read_data_sets('data/mnist', one_hot=False)
  batch_xs, batch_ys = mnist.train.next_batch(TO_EMBED_COUNT)

  # Creating the embeddings
  embedding_var = tf.Variable(batch_xs, name=NAME_TO_VISUALIZE_VARIABLE)
  summary_writer = tf.summary.FileWriter(LOG_DIR)


  # Creating the embedding projectorc
  config = projector.ProjectorConfig()
  embedding = config.embeddings.add()
  embedding.tensor_name = embedding_var.name

  # Specify where you find the metadata
  embedding.metadata_path = path_for_mnist_metadata # 'metadata.tsv'

  # Specify where you find the sprite (we will create this later)
  embedding.sprite.image_path = path_for_mnist_sprites # 'mnistdigits.png'
  embedding.sprite.single_image_dim.extend([28, 28])
  
  # Say that you want to visualize the embeddings
  projector.visualize_embeddings(summary_writer, config)
  

  to_visualize = batch_xs
  to_visualize = vector_to_matrix_mnist(to_visualize)
  to_visualize = invert_grayscale(to_visualize)


  sprite_image = create_sprite_image(to_visualize)
  plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')


  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), 1)

  # Save the metadata
  with open(path_for_mnist_metadata, 'w') as f:
    f.write('Index\tLabel\n')
    for index, label in enumerate(batch_ys):
      f.write('%d\t%d\n' % (index, label))


