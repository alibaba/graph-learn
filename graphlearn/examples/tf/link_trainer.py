# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
"""Distributed trainers for Link Prediction on TensorFlow backend"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time

import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf
from trainer import DistTrainer

from tensorflow.python.client import timeline
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

class LinkDistTrainer(DistTrainer):
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
               save_checkpoint_secs=600,
               save_checkpoint_steps=None,
               profiling=False,
               server_protocol=None):
    super(LinkDistTrainer, self).__init__(cluster_spec, job_name, task_index,
        worker_count, ckpt_dir, save_checkpoint_secs, save_checkpoint_steps,
        profiling, server_protocol)

  def train_and_eval(self, pos_iter, neg_iter, loss,
      test_iter, test_logits, test_neg_iter, test_neg_logits,
      learning_rate=1e-2, epochs=10, hooks=[], hit_K=50):
    with self.context():
      self.global_step = tf.train.get_or_create_global_step()
      try:
        optimizer = tf.train.AdamAsyncOptimizer(learning_rate=learning_rate)
      except AttributeError:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = optimizer.minimize(loss, global_step=self.global_step)
      train_ops = [train_op, loss, self.global_step]
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, pos_iter.initializer)
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, neg_iter.initializer)
      self.init_session(hooks=hooks)
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
          # reinitialize
          self.sess._tf_sess().run(pos_iter.initializer)
          self.sess._tf_sess().run(neg_iter.initializer)
          # eval
          pos_logits = self.eval(test_logits, test_iter)
          neg_logits = self.eval(test_neg_logits, test_neg_iter)
          print('Test hits@{}: {}'.format(hit_K, self.eval_hits(pos_logits, neg_logits, hit_K)))
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

  def predict(self, iterator, logits, src_id, dst_id, writer, batch_size):
    self.save_checkpoint_secs = None
    self.save_checkpoint_steps = None
    with self.context():
      self.global_step = tf.train.get_or_create_global_step()
      if self.sess is None:
        self.init_session()
      local_step = 0
      self.sess.run(iterator.initializer)
      while True:
        try:
          t = time.time()
          outs = self.sess.run([src_id, dst_id, logits])
          writer.write(list(zip(outs[0], outs[1], outs[2])), (0, 1, 2),
                       allow_type_cast=False)  # src_id, dst_id, score
          if local_step % 10 == 0:
            print('Saved {} edges, Time(s) {:.4f}'.format(local_step * batch_size, time.time() - t))
          local_step += 1
        except tf.errors.OutOfRangeError:
          print('Save edges done.')
          break

  def eval(self, logits, iterator):
    self.sess._tf_sess().run(iterator.initializer)
    outs = np.array([])
    while True:
      try:
        outs = np.append(self.sess._tf_sess().run(logits), outs)
      except tf.errors.OutOfRangeError:
        print('End of an epoch.')
        break
    return outs

  def eval_hits(self, y_pred_pos, y_pred_neg, k):
    '''
      compute Hits@K
      For each positive target node, the negative target nodes are the same.
      y_pred_neg is an array.
      rank y_pred_pos[i] against y_pred_neg for each i
    '''
    if len(y_pred_neg) < k or len(y_pred_pos) == 0:
      return {'hits@{}'.format(k): 1.}
    kth_score_in_negative_edges = np.sort(y_pred_neg)[-k]
    hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)
    return {'hits@{}'.format(k): hitsK}

  def join(self):
    self.server.join()