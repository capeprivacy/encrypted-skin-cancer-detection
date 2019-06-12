import numpy as np

import tensorflow as tf
from fastai.vision import *

from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util


def convert_fastai_to_tf(pytorch_model, input):

  has_reshaped = False
  x = input

  for _, m in enumerate(pytorch_model.modules()):
    if isinstance(m, nn.Conv2d):
      p = list(m.parameters())
      x = create_conv2d(x, p[0].data.numpy(), p[1].data.numpy())

    if isinstance(m, nn.BatchNorm2d):
      p = list(m.parameters())
      x = create_batch_norm(x,
                            mean=m.running_mean.numpy(),
                            variance=m.running_var.numpy(),
                            offset=p[1].data.numpy(),
                            scale=p[0].data.numpy(),
                            epsilon=m.eps)

    if isinstance(m, nn.BatchNorm1d):
      p = list(m.parameters())

      x = create_batch_norm(x,
                            m.running_mean.numpy(),
                            m.running_var.numpy(),
                            offset=p[1].data.numpy(),
                            scale=p[0].data.numpy(),
                            epsilon=m.eps)

    if isinstance(m, nn.ReLU):
      x = create_relu(x)

    if isinstance(m, nn.AvgPool2d):
      x = create_avgpool2d(x)

    if isinstance(m, nn.MaxPool2d):
      x = create_maxpool2d(x)

    if isinstance(m, nn.Linear):
      if has_reshaped == False:
        has_reshaped = True
        x = tf.transpose(x, (0, 3, 1, 2))
        x = tf.reshape(x, [-1, np.prod(x.shape.as_list())])
        x = create_linear(x,
                          m.weight.data.numpy().transpose(),
                          m.bias.data.numpy())
      else:
        x = create_linear(x,
                          m.weight.data.numpy().transpose(),
                          m.bias.data.numpy())

  return x


def export_to_pb(pytorch_model, filename):

  tf.reset_default_graph()
  input = tf.placeholder(tf.float32, shape=(1, 200, 200, 3))
  x = convert_fastai_to_tf(pytorch_model, input)

  with tf.Session() as sess:
    pred_node_names = ["output"]
    pred = [tf.identity(x, name=pred_node_names[0])]

    graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        pred_node_names)

    graph = graph_util.remove_training_nodes(graph)

    path = graph_io.write_graph(graph, ".", filename, as_text=False)
    print('saved the frozen graph (ready for inference) at: ', filename)

  return path


def expand_dim(t):
  if t is not None:
    t = np.expand_dims(t, 0)
    t = np.expand_dims(t, 1)
    t = np.expand_dims(t, 2)
  return t


def create_batch_norm(input,
                      mean,
                      variance,
                      offset=None,
                      scale=None,
                      epsilon=0):

  if len(input.shape) == 4:
    mean = expand_dim(mean)
    variance = expand_dim(variance)
    offset = expand_dim(offset)
    scale = expand_dim(scale)

  return tf.nn.batch_normalization(input,
                                   mean,
                                   variance,
                                   offset,
                                   scale,
                                   epsilon)


def create_relu(input):
  return tf.nn.relu(input)


def create_conv2d(input, filter, bias):
  filter = np.transpose(filter, (2, 3, 1, 0))

  x = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1],
                   padding="SAME")

  return tf.nn.bias_add(x, bias)


def create_avgpool2d(input):
  return tf.nn.avg_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")


def create_maxpool2d(input):
  return tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")


def create_linear(input, weights, bias):
  x = tf.nn.bias_add(tf.matmul(input, weights), bias)
  return x
