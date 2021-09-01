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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from graphlearn import pywrap_graphlearn as pywrap
import graphlearn.python.errors as errors

class KnnOption(object):
  def __init__(self, k):
    self.k = k

class KnnOperator(object):
  """ K nearest neighbor operator.
  """

  def __init__(self, client):
    self._client = client

  def search(self, node_type, inputs, k):
    """ Get k nearest neighbors for inputs.

    `inputs`(np.ndarray), with one or two dimensions. If one, it means the
    batch size is 1, and the lengh of the array is vector size.
    If two, it means the batch size is shape[0], and vector size is shape[1].

    Return:
      Two `np.ndarray` objects, `ids` and `distances`, shape=[`batch_size`]
    """
    if not isinstance(inputs, np.ndarray):
      raise ValueError("The knn inputs must be a np.ndarray")

    batch_size = 1
    dimension = -1
    if len(inputs.shape) == 1:
      dimension = inputs.shape[0]
    elif len(inputs.shape) == 2:
      batch_size = inputs.shape[0]
      dimension = inputs.shape[1]
    else:
      raise ValueError("The knn inputs must be with 1 or 2 dimensions")

    req = pywrap.new_knn_request(node_type, k)
    res = pywrap.new_knn_response()
    pywrap.set_knn_request(req, batch_size, dimension, inputs)
    status = self._client.run_op(req, res)
    if status.ok():
      ids = pywrap.get_knn_ids(res)
      distances = pywrap.get_knn_distances(res)
    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    errors.raise_exception_on_not_ok_status(status)

    return np.reshape(ids, [batch_size, k]), \
           np.reshape(distances, [batch_size, k])
