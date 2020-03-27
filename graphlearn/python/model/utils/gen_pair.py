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
"""Function of generating pairs of given path."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def gen_pair(path, left_window_size, right_window_size):
  """
  Args:
    path: a list of ids start with root node's ids, each element is 1d numpy array
    with the same size.
  Returns:
    a pair of numpy array ids.

  Example:
  >>> path = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
  >>> left_window_size = right_window_size = 1
  >>> src_id, dst_ids = gen_pair(path, left_window_size, right_window_size)
  >>> print print(src_ids, dst_ids)
  >>> (array([1, 2, 3, 4, 3, 4, 5, 6]), array([3, 4, 1, 2, 5, 6, 3, 4]))
  """

  path_len = len(path)
  pairs = [[], []] # [src ids list, dst ids list]

  for center_idx in range(path_len):
    cursor = 0
    while center_idx - cursor > 0 and  cursor < left_window_size:
      pairs[0].append(path[center_idx])
      pairs[1].append(path[center_idx - cursor - 1])
      cursor += 1

    cursor = 0
    while center_idx + cursor + 1 < path_len and cursor < right_window_size:
      pairs[0].append(path[center_idx])
      pairs[1].append(path[center_idx + cursor + 1])
      cursor += 1
  return np.concatenate(pairs[0]), np.concatenate(pairs[1])
