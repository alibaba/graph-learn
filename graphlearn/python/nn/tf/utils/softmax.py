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
"""unsorted_segment_softmax op"""

import tensorflow as tf

def unsorted_segment_softmax(data, segment_ids, num_segments):
  """Computes segment_softmax.
  This function first groups the data along the first dimension
  based on segment_ids, and then proceeds to compute the softmax individually
  for each group.

  Args:
    data: A Tensor.
    segment_ids: A Tensor. Must be one of the following types: int32,int64.
      A tensor whose shape is a prefix of data.shape.
    num_segments: A Tensor. Must be one of the following types: int32,int64.
  Returns:
    A Tensor. Has the same type and shape as data.
  """
  max_v = tf.gather(tf.math.unsorted_segment_max(
                      data,
                      segment_ids,
                      num_segments=num_segments),
                    segment_ids)
  # avoids overflow.
  out = tf.exp(data - tf.stop_gradient(max_v))
  out_sum = tf.gather(tf.math.unsorted_segment_sum(
                        out,
                        segment_ids,
                        num_segments=num_segments),
                      segment_ids)
  return out / (out_sum + 1e-10)
