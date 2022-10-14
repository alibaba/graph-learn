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
import sys
import time

import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

import graphlearn as gl
import graphlearn.python.nn.tf as tfg

sys.path.append("../..")
from trainer import DistTrainer as BaseDistTrainer


class DistTrainer(BaseDistTrainer):
  """
  Custom DistTrainer save node embedding on local file.
  """
  def save(self, output, iterator, ids, emb, batch_size):
    print('Start saving embeddings...')
    with self.context():
        local_step = 0
        self.sess._tf_sess().run(iterator.initializer)
        while True:
          try:
            t = time.time()
            outs = self.sess._tf_sess().run([ids, emb])
            # [B,], [B,dim]
            with open(output, 'a') as f:
              feat = [','.join(str(x) for x in arr) for arr in outs[1]]
              for i, e in zip(outs[0], feat):
                f.write("{}\t{}\n".format(i, e))  # id,emb
            if local_step % 10 == 0:
              print('Saved {} node embeddings, Time(s) {:.4f}'.format(local_step * batch_size, time.time() - t))
            local_step += 1
          except tf.errors.OutOfRangeError:
            print('Save node embeddings done.')
            break
      # Prevent chief worker from exiting before other workers start.
      # if self.task_index == 0:
      #   time.sleep(60 * 2)
      # print('Write embedding done!')
