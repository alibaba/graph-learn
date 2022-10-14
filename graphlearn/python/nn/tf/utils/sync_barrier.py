# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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
"""sync barrier for distributed training."""

import logging
import time

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf


class SyncBarrierHook(tf.train.SessionRunHook):
  def __init__(self, num_worker, is_chief):
    self._num_worker = num_worker
    self._is_chief = is_chief
    self._queue_size = None
    self._queue = None
    self._enqueue = None
    self._dequeue = None
    self._end = False

  def begin(self):
    """Setup barrier queue.
    """
    logging.info('Number of workers: %d' % self._num_worker)
    with tf.device(tf.DeviceSpec(job='ps', task=0,
                                 device_type='CPU', device_index=0)):
      self._queue = tf.FIFOQueue(capacity=self._num_worker,
                                 dtypes=[tf.float32], shapes=[()],
                                 name='sync_barrier_queue',
                                 shared_name='sync_barrier_queue')
    self._enqueue = self._queue.enqueue(1.0)
    self._queue_size = self._queue.size()
    self._dequeue = self._queue.dequeue()

  def after_create_session(self, session, coord=None):
    """Clean up the queue.
    """
    if self._is_chief:
      queue_size = session.run(self._queue_size)
      while queue_size > 0:
        session.run(self._dequeue)
        queue_size = session.run(self._queue_size)
      logging.info('SyncBarrier queue cleared: %d' % queue_size)

  def end(self, session):
    if not self._end:
      session.run(self._enqueue)
      queue_size = session.run(self._queue_size)
      while queue_size < self._num_worker:
        queue_size = session.run(self._queue_size)
        time.sleep(5)
        logging.info('Waiting for other worker, finished %d, total %d' %
              (queue_size, self._num_worker))
      logging.info('SyncBarrier passed.')
      self._end = True
