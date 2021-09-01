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
"""Distributed trainers on TensorFlow backend"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time

import graphlearn as gl
import graphlearn.python.nn.tf as tfg
import numpy as np
import tensorflow as tf


class DistTrainer(object):
  """Class for distributed training and evaluation

  Args:
    cluster_spec: TensorFlow ClusterSpec.
    job_name: name of this worker.
    task_index: index of this worker.
    worker_count: The number of TensorFlow worker.
    ckpt_dir: checkpoint dir.
  """
  def __init__(self,
               cluster_spec,
               job_name,
               task_index,
               worker_count,
               ckpt_dir=None):
    self.cluster_spec = cluster_spec
    self.job_name = job_name
    self.task_index = task_index
    self.worker_count = worker_count
    self.ckpt_dir = ckpt_dir

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = False
    conf.device_filters.append('/job:ps')
    conf.device_filters.append('/job:worker/task:%d' % self.task_index)
    conf.inter_op_parallelism_threads = 1
    self.server = tf.train.Server(self.cluster_spec,
                                  job_name=self.job_name,
                                  task_index=self.task_index,
                                  config=conf)
    self.context = self.context
    self.sync_barrier = tfg.SyncBarrierHook(self.worker_count, self.task_index == 0)

  def __exit__(self, exc_type, exc_value, tracebac):
    if self.sess:
      self.sess.close()
    return True

  def context(self):
    return tf.device(tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % self.task_index,
        cluster=self.cluster_spec))

  def init_session(self):
    hooks = [self.sync_barrier]
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = False
    conf.device_filters.append('/job:ps')
    conf.device_filters.append('/job:worker/task:%d' % self.task_index)
    conf.inter_op_parallelism_threads = 1
    if self.ckpt_dir is not None:
      self.sess = tf.train.MonitoredTrainingSession(
          master=self.server.target,
          checkpoint_dir=self.ckpt_dir,
          save_checkpoint_secs=1800,
          is_chief=(self.task_index == 0),
          hooks=hooks,
          config=conf)
    else:
      self.sess = tf.train.MonitoredTrainingSession(
          master=self.server.target,
          is_chief=(self.task_index == 0),
          hooks=hooks,
          config=conf)

  def train(self, iterator, loss, learning_rate, epochs=10, **kwargs):
    with self.context():
      self.global_step = tf.train.get_or_create_global_step()
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = optimizer.minimize(loss, global_step=self.global_step)
      # if self._use_input_bn:
      #   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      #   train_op = tf.group([train_op, update_ops])
      train_ops = [train_op, loss,
                   self.global_step]
      self.init_session()
      print('Start training...')
      local_step = 0
      t = time.time()
      last_global_step = 0
      epoch = 0
      self.sess._tf_sess().run(iterator.initializer)
      while not self.sess.should_stop():
        try:
          outs = self.sess.run(train_ops)
        except tf.errors.OutOfRangeError:
          epoch += 1
          print('End of an epoch.')
          self.sess._tf_sess().run(iterator.initializer)
          if epoch >= epochs:
            break
          else:
            continue
        train_loss = outs[1]
        global_step = outs[-1]
        # Print results
        if local_step % 10 == 0:
          print(datetime.datetime.now(),
                'Epoch {}, Iter {}, Global_step/sec {:.2f}, Time(s) {:.4f}, '
                'Loss {:.5f}'
                .format(epoch, local_step,
                        (global_step - last_global_step) * 1.0 / (time.time() - t),
                        (time.time() - t) * 1.0 / 10,
                        train_loss))
          t = time.time()
          last_global_step = global_step
        local_step += 1
      self.sync_barrier.end(self.sess)

  def join(self):
    self.server.join()
