# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Local and distributed trainers on TensorFlow backend"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.model.trainer import Trainer


class TFTrainer(Trainer):
  """Base class of TF trainer

  Args:
    model_func: A model instance.
    epoch: training epochs.
    optimizer: A tf optimizer instance.
  """

  def __init__(self,
               model_func,
               epoch=100,
               optimizer=tf.train.AdamOptimizer()):
    super(TFTrainer, self).__init__(model_func, epoch, optimizer)
    if not self._model_func:
      raise NotImplementedError('model_func to be implemented.')
    self.model = None
    self.sess = None

  def __exit__(self, exc_type, exc_value, tracebac):
    if self.sess:
      self.sess.close()
    return True

  def init(self, **kwargs):
    raise NotImplementedError('TFTrainer init() to be implemented.')

  def get_node_embedding(self, node_type=None):
    print('save embedding...')
    ids, emb, iterator = self.model.node_embedding(node_type)
    emb_set = []
    feed_dict = self.model.feed_evaluation_args()
    if iterator is not None:
      self.sess.run(iterator.initializer)
      while True:
        try:
          ids_np, emb_np = self.sess.run([ids, emb], feed_dict=feed_dict)
          emb_set.append(np.concatenate([np.reshape(ids_np, [-1, 1]), emb_np],
                                        axis=1))
        except tf.errors.OutOfRangeError:
          break
    else: # sample full graph in one batch
      ids_np, emb_np = self.sess.run([ids, emb], feed_dict=feed_dict)
      emb_set.append(np.concatenate([np.reshape(ids_np, [-1, 1]), emb_np],
                                    axis=1))
    return np.concatenate(emb_set, axis=0)

  def get_node_embedding_fixed(self, node_type=None):
    print('save embedding...')
    ids, emb, iterator = self.model.node_embedding(node_type)
    feed_dict = self.model.feed_evaluation_args()
    emb_ids = []
    emb_values = []
    if iterator is not None:
      self.sess.run(iterator.initializer)
      while True:
        try:
          ids_np, emb_np = self.sess.run([ids, emb], feed_dict=feed_dict)
          # emb_set.append(np.concatenate([np.reshape(ids_np, [-1, 1]), emb_np],
          #                               axis=1))
          emb_ids.append(ids_np)
          emb_values.append(emb_np)
        except tf.errors.OutOfRangeError:
          break
    else: # sample full graph in one batch
      ids_np, emb_np = self.sess.run([ids, emb], feed_dict=feed_dict)
      # emb_set.append(np.concatenate([np.reshape(ids_np, [-1, 1]), emb_np],
      #                               axis=1))
      emb_ids.append(ids_np)
      emb_values.append(emb_np)
    # return np.concatenate(emb_set, axis=0)
    return np.concatenate(emb_ids), np.concatenate(emb_values)

  def _train_epoch(self, train_op, loss, iterator, idx):
    """train one epoch with given dataset(graph)"""
    total_loss = []
    dur = []
    feed_dict = self.model.feed_training_args()
    if iterator is not None:
      self.sess.run(iterator.initializer)
      iter = 0
      while True:
        try:
          start_time = time.time()
          outs = self.sess.run([train_op, loss],
                               feed_dict=feed_dict)
          end_time = time.time()
          total_loss.append(outs[1])
          iter_time = end_time - start_time
          dur.append(iter_time)
          print("Epoch {:02d}, Iteration {}, Time(s) {:.4f}, Loss {:.5f}"
                .format(idx, iter, iter_time, outs[1]))
          iter += 1
        except tf.errors.OutOfRangeError:
          break
    else:  # sample full graph in one batch
      start_time = time.time()
      outs = self.sess.run([train_op, loss],
                           feed_dict=feed_dict)
      end_time = time.time()
      total_loss.append(outs[1])
      dur.append(end_time - start_time)
      print("Epoch {:02d}, Time(s) {:.4f}, Loss {:.5f}"
            .format(idx, np.sum(dur), np.mean(total_loss)))

  def _evaluate(self, mode, acc, iterator):
    """evaluate accuracy"""
    total_acc = []
    feed_dict = self.model.feed_evaluation_args()
    if iterator is not None:
      self.sess.run(iterator.initializer)
      while True:
        try:
          total_acc.append(self.sess.run(acc, feed_dict=feed_dict))
        except tf.errors.OutOfRangeError:
          break
    else: # sample full graph in one batch
      total_acc.append(self.sess.run(acc, feed_dict=feed_dict))
    print('{} Accuracy is: {:.4f}'.format(mode, np.mean(total_acc)))


class LocalTFTrainer(TFTrainer):
  """Class for local training and evaluation."""
  def __init__(self,
               model_func,
               epoch=100,
               optimizer=tf.train.AdamOptimizer()):
    super(LocalTFTrainer, self).__init__(model_func,
                                         epoch,
                                         optimizer)
    self.model = self._model_func()

  def init(self, **kwargs):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    self.sess.run(tf.local_variables_initializer())
    self.sess.run(tf.global_variables_initializer())

  def train(self, **kwargs):
    """tf.worker method.
    """
    loss, train_iterator = self.model.build()
    self.global_step = tf.train.get_or_create_global_step()
    train_op = self._optimizer.minimize(loss, global_step=self.global_step)

    self.init()
    print('training...')
    for i in range(self._epoch):
      self._train_epoch(train_op, loss, train_iterator, i)

  def evaluate(self, mode='val', **kwargs):
    if mode=='val':
      acc = self.model.val_acc
      iterator = self.model.val_iterator
    else:
      acc = self.model.test_acc
      iterator = self.model.test_iterator

    self.init()
    self._evaluate(mode, acc, iterator)

  def train_and_evaluate(self, **kwargs):
    loss, train_iterator = self.model.build()
    self.global_step = tf.train.get_or_create_global_step()
    train_op = self._optimizer.minimize(loss, global_step=self.global_step)
    val_acc, val_iterator = self.model.val_acc()
    test_acc, test_iterator = self.model.test_acc()

    self.init()
    print('training...')
    for i in range(self._epoch):
      self._train_epoch(train_op, loss, train_iterator, i)
      self._evaluate('val', val_acc, val_iterator)
    print('test ...')
    self._evaluate('test', test_acc, test_iterator)


class DistTFTrainer(TFTrainer):
  """Class for distributed training and evaluation

  Args:
    model_func: A model instance.
    cluster_spec: Spec used to describe the distribute cluster.
    task_name: name of this worker.
    task_index: index of this worker.
    epoch: training epochs, default is 100.
    lr: learning rate.
    optimizer: Optimizer methods.
    weight_decay: weight decay rate.
    val_frequency: validation frequency, default is 10.
  """
  def __init__(self,
               model_func,
               cluster_spec,
               task_name,
               task_index,
               epoch=100,
               optimizer=tf.train.AdamOptimizer(),
               val_frequency=10):
    super(DistTFTrainer, self).__init__(model_func,
                                        epoch,
                                        optimizer)
    self.cluster_spec = cluster_spec
    self.task_name = task_name
    self.task_index = task_index
    self._val_frequency = val_frequency

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = False
    self.server = tf.train.Server(self.cluster_spec,
                                  job_name=self.task_name,
                                  task_index=self.task_index,
                                  config=conf)
    self.context = self.context
    with self.context():
      self.model = self._model_func()

  def init(self, **kwargs):
    self.sess = tf.train.MonitoredTrainingSession(
        master=self.server.target,
        is_chief=(self.task_index == 0))

  def context(self, **kwargs):
    return tf.device(tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % self.task_index,
        cluster=self.cluster_spec))

  def train(self, **kwargs):
    with self.context(**kwargs):
      loss, train_iterator = self.model.build()
      self.global_step = tf.train.get_or_create_global_step()
      train_op = self._optimizer.minimize(loss, global_step=self.global_step)

    self.init()
    print('training...')
    for i in range(self._epoch):
      self._train_epoch(train_op, loss, train_iterator, i)

  def evaluate(self, mode='val', **kwargs):
    with self.context(**kwargs):
      if mode=='val':
        acc = self.model.val_acc
        iterator = self.model.val_iterator
      else:
        acc = self.model.test_acc
        iterator = self.model.test_iterator
    self.init()
    self._evaluate(mode, acc, iterator)

  def train_and_evaluate(self, **kwargs):
    with self.context(**kwargs):
      loss, train_iterator = self.model.build()
      self.global_step = tf.train.get_or_create_global_step()
      train_op = self._optimizer.minimize(loss, global_step=self.global_step)
      val_acc, val_iterator = self.model.val_acc()
      test_acc, test_iterator = self.model.test_acc()

    self.init()
    print('training...')
    for i in range(self._epoch):
      self._train_epoch(train_op, loss, train_iterator, i)
      if (i % self._val_frequency) == 0:
        self._evaluate('val', val_acc, val_iterator)
    print('test ...')
    self._evaluate('test', test_acc, test_iterator)

  def join(self):
    self.server.join()
