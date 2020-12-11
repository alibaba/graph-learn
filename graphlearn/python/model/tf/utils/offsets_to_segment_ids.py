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
"""Transfor offsets to segment_ids, used for GCN and GAT"""
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf


def offsets_to_segment_ids(offsets):
  '''Transforms offsets to segment_ids,
  the segment_ids will be used in tf.segment_sum/segment_mean
  [3, 0, 1, 2] -> [0, 0, 0, 1, 3, 3].
  '''
  c = tf.cumsum(offsets)
  return tf.searchsorted(c, tf.range(c[-1]), side='right')
