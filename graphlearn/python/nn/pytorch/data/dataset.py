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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import os

try:
  import torch as th
except ImportError:
  pass

from graphlearn.python.errors import OutOfRangeError
from graphlearn.python.nn.dataset import Dataset as RawDataset
from graphlearn.python.nn.pytorch.data.utils import get_cluster_spec, is_server_launched

class Dataset(th.utils.data.IterableDataset):
  def __init__(self, query, window=10, induce_func=None, graph=None, cluster=None):
    """Dataset reformats the sampled batch from GSL query as `Data` object
    consists of Pytorch Tensors.
    Args:
      query: a GSL query.
      window: the buffer size for query.
      induce_func:  A function that takes in the query result `Data` dict and
        returns a list of subgraphs. For pyG, these subgraphs are pyG `Data`
        objects.
    """
    self._query = query
    self._window = window
    self._cluster = cluster

    self._graph = graph
    self._lazy_init = self._graph is not None
    if not self._lazy_init:
      self._rds = RawDataset(self._query, window=self._window)
    else:
      if not is_server_launched():
        raise RuntimeError('graph learn server should be running firstly when using lazy init dataset')
      if self._cluster is None:
        self._cluster = get_cluster_spec()
      self._rds = None

    self._client_id = 0
    self._format = lambda x: x
    self._induce_func = induce_func

  def __iter__(self):
    if self._lazy_init and self._rds is None:
      self._init_raw_dataset()
      atexit.register(self._graph.close)
    def iterator():
      while True:
        try:
          value = self._rds.get_data_dict()
          if self._induce_func is not None:
            value = self._induce_func(value) # get a list of pyG Data objects.
          else:
            for k, v in value.items():
              v.apply(self._convert_func)
              value[k] = self._format(v)
          yield value
        except OutOfRangeError:
          break
    return iterator()

  def as_dict(self):
    """Convert each `Data` to dict of torch tensors.
    This function is used for raw `DataLoader` of pytorch.
    """
    def func(x):
      return {k: v for k, v in x.__dict__.items() if v is not None}
    self._format = func
    return self

  def lazy_init(self):
    return self._lazy_init

  def _convert_func(self, data):
    res = {}
    if isinstance(data, dict):
      for k, v in data.items():
        res[k] = self._convert_func(v)
      return res
    return th.from_numpy(data)

  def _init_raw_dataset(self):
    self._graph.init(cluster=self._cluster, job_name="client", task_index=self.client_id)
    self._rds = RawDataset(self._query, window=self._window)

  @property
  def client_id(self):
    return self._client_id

  @client_id.setter
  def client_id(self, value):
    if not isinstance(value, int):
      raise ValueError('client_id should be int type')
    self._client_id = value
