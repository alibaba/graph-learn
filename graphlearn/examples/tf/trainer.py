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

import atexit
import datetime
import os
import time

import numpy as np
import graphlearn as gl
import graphlearn.python.nn.tf as tfg

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from tensorflow.python.client import timeline
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

class DistTrainer(object):
  """Class for distributed training and evaluation

  Args:
    cluster_spec: TensorFlow ClusterSpec.
    job_name: name of this worker.
    task_index: index of this worker.
    worker_count: The number of TensorFlow worker.
    ckpt_dir: checkpoint dir.
    profiling: Whether write timeline for profiling, default is False.
  """
  def __init__(self,
               cluster_spec,
               job_name,
               task_index,
               worker_count,
               ckpt_dir=None,
               ckpt_freq=None,
               profiling=False):
    self.cluster_spec = cluster_spec
    self.job_name = job_name
    self.task_index = task_index
    self.worker_count = worker_count
    self.ckpt_dir = ckpt_dir
    self.ckpt_freq = ckpt_freq
    self.profiling = profiling

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = False
    conf.device_filters.append('/job:ps')
    conf.device_filters.append('/job:worker/task:%d' % self.task_index)
    #conf.inter_op_parallelism_threads = 1
    self.server = tf.train.Server(self.cluster_spec,
                                  job_name=self.job_name,
                                  task_index=self.task_index,
                                  config=conf)
    self.context = self.context
    self.sync_barrier = tfg.SyncBarrierHook(self.worker_count, self.task_index == 0)
    self.sess = None

  def context(self):
    return tf.device(tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % self.task_index,
        cluster=self.cluster_spec))

  def init_session(self, hooks=[]):
    hooks_ = [self.sync_barrier]
    hooks_.extend(hooks)
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = False
    conf.device_filters.append('/job:ps')
    conf.device_filters.append('/job:worker/task:%d' % self.task_index)
    #conf.inter_op_parallelism_threads = 1
    if self.ckpt_dir is not None:
      self.sess = tf.train.MonitoredTrainingSession(
          master=self.server.target,
          checkpoint_dir=self.ckpt_dir,
          save_checkpoint_secs=self.ckpt_freq,
          is_chief=(self.task_index == 0),
          hooks=hooks_,
          config=conf)
    else:
      self.sess = tf.train.MonitoredTrainingSession(
          master=self.server.target,
          is_chief=(self.task_index == 0),
          hooks=hooks_,
          config=conf)

    def _close_session():
      if self.sess is not None:
        self.sess.close()
    atexit.register(_close_session)

  def train(self, iterator, loss, learning_rate, epochs=10, hooks=[], **kwargs):
    with self.context():
      self.global_step = tf.train.get_or_create_global_step()
      try:
        optimizer = tf.train.AdamAsyncOptimizer(learning_rate=learning_rate)
      except AttributeError:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = optimizer.minimize(loss, global_step=self.global_step)
      # if self._use_input_bn:
      #   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      #   train_op = tf.group([train_op, update_ops])
      train_ops = [train_op, loss, self.global_step]
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
      self.init_session(hooks=hooks)
      print('Start training...')
      local_step = 0
      t = time.time()
      last_global_step = 0
      epoch = 0
      outs = None
      while (not self.sess.should_stop()) and (epoch < epochs):
        try:
          if self.profiling and self.task_index == 1 and \
            local_step % 100 == 0 and local_step > 500 and local_step < 1000:
            outs = self.sess.run(train_ops,
                                 options=run_options,
                                 run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            content = tl.generate_chrome_trace_format()
            file_name = 'timeline_' + str(local_step) + '_' + \
              str(self.task_index) + '.json'
            save_path = os.path.join(self.ckpt_dir, file_name)
            writeGFile = tf.gfile.GFile(save_path, mode='w')
            writeGFile.write(content)
            writeGFile.flush()
            writeGFile.close()
            print("Profiling data save to %s success." % save_path)
          else:
            outs = self.sess.run(train_ops)
        except tf.errors.OutOfRangeError:
          epoch += 1
          print('End of an epoch.')
          self.sess._tf_sess().run(iterator.initializer)
        if outs is not None:
          train_loss = outs[1]
          global_step = outs[-1]
          # Print results
          if local_step % 10 == 0:
            print(datetime.datetime.now(),
                  'Epoch {}, Iter {}, Global_step/sec {:.2f}, Time(s) {:.4f}, '
                  'Loss {:.5f}, Global_step {}'
                  .format(epoch, local_step,
                          (global_step - last_global_step) * 1.0 / (time.time() - t),
                          (time.time() - t) * 1.0 / 10,
                          train_loss, global_step))
            t = time.time()
            last_global_step = global_step
          local_step += 1

      self.sync_barrier.end(self.sess)

  def save_node_embedding(self, emb_writer, iterator, ids, emb, batch_size):
    print('Start saving embeddings...')
    with self.context():
      self.global_step = tf.train.get_or_create_global_step()
      if self.sess is None:
        self.init_session()
      local_step = 0
      self.sess._tf_sess().run(iterator.initializer)
      while True:
        try:
          t = time.time()
          outs = self.sess._tf_sess().run([ids, emb])
          # [B,], [B,dim]
          feat = [','.join(str(x) for x in arr) for arr in outs[1]]
          emb_writer.write(list(zip(outs[0], feat)), indices=[0, 1])  # id,emb
          if local_step % 10 == 0:
            print('Saved {} node embeddings, Time(s) {:.4f}'.format(local_step * batch_size, time.time() - t))
          local_step += 1
        except tf.errors.OutOfRangeError:
          print('Save node embeddings done.')
          break
      # Prevent chief worker from exiting before other workers start.
      # if self.task_index == 0:
      #   time.sleep(60 * 2)
      # print('Write to ODPS table done!')

  def join(self):
    self.server.join()
