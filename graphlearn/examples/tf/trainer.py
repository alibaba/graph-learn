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
import contextlib
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


class TFTrainer(object):
  """Class for local or distributed training and evaluation.

  Args:
    ckpt_dir: checkpoint dir.
    save_checkpoint_secs: checkpoint frequency.
    save_checkpoint_steps: checkpoint steps.
    profiling: whether write timeline for profiling, default is False.
    progress_steps: print a progress logs for given steps.
  """
  def __init__(self,
               ckpt_dir=None,
               save_checkpoint_secs=600,
               save_checkpoint_steps=None,
               profiling=False,
               progress_steps=10):
    self.ckpt_dir = ckpt_dir
    self.save_checkpoint_secs = save_checkpoint_secs
    self.save_checkpoint_steps = save_checkpoint_steps
    self.profiling = profiling
    self.progress_steps = progress_steps

    self.conf = tf.ConfigProto()
    self.conf.gpu_options.allow_growth = True
    self.conf.allow_soft_placement = False
    self.sess = None

    # use for distributed training
    self.sync_barrier = None
    self.global_step = None
    self.is_local = None

  def context(self):
    raise NotImplementedError('Use LocalTrainer or DistTrainer instead.')

  def init_session(self, hooks=None, **kwargs):
    if isinstance(hooks, (list, tuple)):
      hooks_ = [hook for hook in hooks]
    elif hooks is not None:
      hooks_ = [hooks]

    checkpoint_args = dict()
    if self.ckpt_dir is not None:
      checkpoint_args['checkpoint_dir'] = self.ckpt_dir
    if self.save_checkpoint_secs is not None:
      checkpoint_args['save_checkpoint_secs'] = self.save_checkpoint_secs
    if self.save_checkpoint_steps is not None:
      checkpoint_args['save_checkpoint_steps'] = self.save_checkpoint_steps

    self.sess = tf.train.MonitoredTrainingSession(
        hooks=hooks_,
        config=self.conf,
        **checkpoint_args,
        **kwargs)

    def _close_session():
      if self.sess is not None:
        self.sess.close()
    atexit.register(_close_session)

  def run_step(self, train_ops, local_step):
    raise NotImplementedError('Use LocalTrainer or DistTrainer instead.')

  def train(self, iterator, loss, optimizer=None, learning_rate=None,
            epochs=10, hooks=[], **kwargs):
    with self.context():
      self.global_step = tf.train.get_or_create_global_step()
      if optimizer is None:
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
      last_local_step = 0
      last_global_step = 0
      t = time.time()
      epoch = 0
      outs = None
      while (not self.sess.should_stop()) and (epoch < epochs):
        try:
          outs = self.run_step(train_ops, local_step)
        except tf.errors.OutOfRangeError:
          print('End of the epoch %d.' % (epoch,))
          epoch += 1
          self.sess._tf_sess().run(iterator.initializer)  # reinitialize dataset.
        if outs is not None:
          train_loss = outs[1]
          global_step = outs[-1]
          # Print results
          local_step += 1
          if local_step % self.progress_steps == 0:
            if self.is_local:
              print(datetime.datetime.now(),
                    'Epoch {}, Iter {}, LocalStep/sec {:.2f}, Time(s) {:.4f}, '
                    'Loss {:.5f}'
                    .format(epoch, local_step,
                            (local_step - last_local_step) * 1.0 / (time.time() - t),
                            (time.time() - t) * 1.0 / 10, train_loss))
            else:
              print(datetime.datetime.now(),
                    'Epoch {}, Iter {}, GlobalStep/sec {:.2f}, Time(s) {:.4f}, '
                    'Loss {:.5f}, Global_step {}'
                    .format(epoch, local_step,
                            (global_step - last_global_step) * 1.0 / (time.time() - t),
                            (time.time() - t) * 1.0 / 10, train_loss, global_step))
            t = time.time()
            last_local_step = local_step
            last_global_step = global_step

      if self.sync_barrier is not None:
        self.sync_barrier.end(self.sess)

  def test(self, iterator, test_acc, hooks=[], **kwargs):
    with self.context():
      self.global_step = tf.train.get_or_create_global_step()
      if self.sess is None:
        self.init_session(hooks=hooks)

      print('Start testing ...')
      total_test_acc = []
      local_step = 0
      last_local_step = 0
      last_global_step = 0
      self.sess._tf_sess().run(iterator.initializer)
      try:
        while True:
          t = time.time()
          outs = self.sess._tf_sess().run([test_acc, self.global_step])
          if outs is not None:
            accuracy = outs[0]
            global_step = outs[-1]
            local_step += 1
            # Print results
            if local_step % self.progress_steps == 0:
              if self.is_local:
                print(datetime.datetime.now(),
                      'Iter {}, LocalStep/sec {:.2f}, Time(s) {:.4f}, '
                      'Accuracy {:.5f}'
                      .format(local_step,
                              (local_step - last_local_step) * 1.0 / (time.time() - t),
                              (time.time() - t) * 1.0 / 10, accuracy))
              else:
                print(datetime.datetime.now(),
                      'Iter {}, GlobalStep/sec {:.2f}, Time(s) {:.4f}, '
                      'Accuracy {:.5f}, Global_step {}'
                      .format(local_step,
                              (global_step - last_global_step) * 1.0 / (time.time() - t),
                              (time.time() - t) * 1.0 / 10, accuracy, global_step))
              t = time.time()
              last_local_step = local_step
              last_global_step = global_step
          total_test_acc.append(accuracy)
      except tf.errors.OutOfRangeError:
        print("Finished.")
      print('Test Accuracy is: {:.4f}'.format(np.mean(total_test_acc)))

  def train_and_evaluate(self, train_iterator, test_iterator, loss, test_acc, optimizer=None, learning_rate=None,
                         epochs=10, hooks=[], **kwargs):
    self.train(train_iterator, loss, optimizer, learning_rate, epochs, hooks, **kwargs)
    self.test(test_iterator, test_acc, hooks, **kwargs)

  def save_node_embedding_bigdata(self, save_iter, save_ids, save_emb, save_file, block_max_lines, batch_size):
    if batch_size >= block_max_lines:
      emb_writer = open(save_file, 'w')
      emb_writer.write('id:int64\temb:string\n')
      self.save_node_embedding(emb_writer, save_iter, save_ids, save_emb, batch_size)
    else:
      print('Start saving embeddings...')
      with self.context():
        self.global_step = tf.train.get_or_create_global_step()
        if self.sess is None:
          self.init_session()
        self.sess._tf_sess().run(save_iter.initializer)
        total_line = 0
        block_id = 0
        save_file = save_file[0:-4] if save_file.endswith('.txt') else save_file
        current_file = save_file + '_%d.txt' % block_id
        with open(current_file, 'a') as f:
          f.write('id:int64\temb:string\n')        
        while True:
          try:
            outs = self.sess._tf_sess().run([save_ids, save_emb])
            id_feat = ['%d\t'%i + ','.join(str(x) for x in arr) + '\n' for i, arr in zip(outs[0], outs[1])]
            if total_line + len(id_feat) <= (block_id + 1) * block_max_lines:
              with open(current_file, 'a') as f:
                f.writelines(id_feat)
              if total_line + len(id_feat) == (block_id + 1) * block_max_lines:
                current_file = save_file + '_%d.txt' % (block_id + 1)
                with open(current_file, 'a') as f:
                  f.write('id:int64\temb:string\n')
                block_id += 1
            elif (block_id + 1) * block_max_lines < total_line + len(id_feat):
              with open(current_file, 'a') as f:
                f.writelines(id_feat[0: (block_id + 1) * block_max_lines - total_line])
              current_file = save_file + '_%d.txt' % (block_id + 1)
              with open(current_file, 'a') as f:
                f.write('id:int64\temb:string\n')
                f.writelines(id_feat[(block_id + 1) * block_max_lines - total_line:])
              block_id += 1
            total_line += len(id_feat)
          except tf.errors.OutOfRangeError:
            print('Save node embeddings done.')
            break
        print("#################################################")
        print("total lines saved = {} , number blocks = {} ".format(total_line, block_id + 1))
        print("#################################################")

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
          id_feat = ['%d\t'%i + ','.join(str(x) for x in arr) + '\n' for i, arr in zip(outs[0], outs[1])]
          emb_writer.writelines(id_feat)  # id,emb
          local_step += 1
          if local_step % self.progress_steps == 0:
            print('Saved {} node embeddings, Time(s) {:.4f}'.format(local_step * batch_size, time.time() - t))
        except tf.errors.OutOfRangeError:
          print('Save node embeddings done.')
          break

class LocalTrainer(TFTrainer):
  """Class for local training and evaluation

  Args:
    ckpt_dir: checkpoint dir.
    save_checkpoint_freq: checkpoint frequency.
    save_checkpoint_steps: checkpoint steps.
    profiling: whether write timeline for profiling, default is False.
    progress_steps: print a progress logs for given steps.
  """
  def __init__(self,
               ckpt_dir=None,
               save_checkpoint_secs=None,
               save_checkpoint_steps=None,
               profiling=False,
               progress_steps=10):
    super().__init__(ckpt_dir, save_checkpoint_secs, save_checkpoint_steps, profiling, progress_steps)
    self.is_local = True 

  if hasattr(contextlib, 'nullcontext'):
    def context(self):
      return contextlib.nullcontext()
  else:
    @contextlib.contextmanager
    def context(self, enter_result=None):
        yield enter_result

  def run_step(self, train_ops, local_step):
    if self.profiling and local_step % 100 == 0 and local_step > 500 and local_step < 1000:
      outs = self.sess.run(train_ops,
                           options=run_options,
                           run_metadata=run_metadata)
      tl = timeline.Timeline(run_metadata.step_stats)
      content = tl.generate_chrome_trace_format()
      file_name = 'timeline_' + str(local_step) + '.json'
      save_path = os.path.join(self.ckpt_dir, file_name)
      writeGFile = tf.gfile.GFile(save_path, mode='w')
      writeGFile.write(content)
      writeGFile.flush()
      writeGFile.close()
      print("Profiling data save to %s success." % save_path)
    else:
      outs = self.sess.run(train_ops)
    return outs

class DistTrainer(TFTrainer):
  """Class for distributed training and evaluation

  Args:
    cluster_spec: TensorFlow ClusterSpec.
    job_name: name of this worker.
    task_index: index of this worker.
    worker_count: The number of TensorFlow worker.
    ckpt_dir: checkpoint dir.
    save_checkpoint_freq: checkpoint frequency.
    save_checkpoint_steps: checkpoint steps.
    profiling: whether write timeline for profiling, default is False.
    progress_steps: print a progress logs for given steps.
  """
  def __init__(self,
               cluster_spec,
               job_name,
               task_index,
               worker_count,
               ckpt_dir=None,
               save_checkpoint_secs=None,
               save_checkpoint_steps=None,
               profiling=False,
               progress_steps=10):
    super().__init__(ckpt_dir, save_checkpoint_secs, save_checkpoint_steps, profiling, progress_steps)
    self.is_local = False
    self.cluster_spec = cluster_spec
    self.job_name = job_name
    self.task_index = task_index
    self.worker_count = worker_count

    self.conf.device_filters.append('/job:ps')
    self.conf.device_filters.append('/job:worker/task:%d' % self.task_index)
    self.server = tf.train.Server(self.cluster_spec,
                                  job_name=self.job_name,
                                  task_index=self.task_index,
                                  config=self.conf)
    self.sync_barrier = tfg.SyncBarrierHook(self.worker_count, self.task_index == 0)

  def context(self):
    return tf.device(tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % self.task_index,
        cluster=self.cluster_spec))

  def init_session(self, hooks=None):
    hooks_ = [self.sync_barrier]
    if isinstance(hooks, (list, tuple)):
      hooks_.extend(hooks)
    elif hooks is not None:
      hooks_.append(hooks)

    self.conf.device_filters.append('/job:ps')
    self.conf.device_filters.append('/job:worker/task:%d' % self.task_index)

    super().init_session(
      hooks=[self.sync_barrier] + (hooks or []),
      master=self.server.target,
      is_chief=(self.task_index == 0),
    )

  def run_step(self, train_ops, local_step):
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
    return outs

  def join(self):
    self.server.join()
