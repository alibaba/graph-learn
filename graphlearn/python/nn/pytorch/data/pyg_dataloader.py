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
"""DataLoader for pyG."""

import os
import torch


try:
  from torch_geometric.data import Data, Batch
except ImportError:
  pass

try:
  from torch_geometric.data import HeteroData # PyG 2.x
except ImportError:
  pass

from torch.utils.data.dataloader import DataLoader, default_collate

from graphlearn.python.nn.pytorch.data.dataset import Dataset as GLDataset
from graphlearn.python.nn.pytorch.data.utils import get_rank, get_num_client

def worker_init_fn(worker_id):
  worker_info = torch.utils.data.get_worker_info()
  dataset = worker_info.dataset
  dataset.client_id = worker_id + get_rank() * get_num_client()

class Collater(object):
  def __init__(self):
    pass
  
  def collate(self, batch):
    """
    Args:
      batch: a list of pyG `Data`.
    """
    batch = batch[0]
    elem = batch[0]
    if isinstance(elem, Data):
      return Batch.from_data_list(batch)
    elif isinstance(elem, torch.Tensor):
      return default_collate(batch)
    elif isinstance(elem, HeteroData):
      return Batch.from_data_list(batch)
    raise TypeError('PyGDataLoader found invalid type: {}'.format(type(elem)))

  def __call__(self, batch):
    return self.collate(batch)


class PyGDataLoader():
    """pyG Data loader which needs specified length.

    Args:
      dataset (Dataset): The dataset to convert GSL and induce a list of pyG `Data` objects.
      length (int): The length presented to the caller,
        beyond which values are discarded to ensure correct server state.
      multi_process (bool): Weather or not use multi-process dataloader
    """
    def __init__(self, dataset, length=None, multi_process=False, **kwargs):
      assert isinstance(dataset, GLDataset), 'PyGDataLoader only accepts GraphLearn datasets'

      self._length = length
      self._index = 0
      self._multi_process = multi_process

      if "batch_size" in kwargs:
        del kwargs["batch_size"]
      if "collate_fn" in kwargs:
        del kwargs["collate_fn"]
      if self._multi_process:
        kwargs['worker_init_fn'] = worker_init_fn
        kwargs['persistent_workers'] = True
        kwargs['num_workers'] = get_num_client()
        if not dataset.lazy_init():
          raise RuntimeError('PyGDataLoader needs a dataset with lazy init')
      self._dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=Collater(), **kwargs)
      self._iterator = None
    
    def __iter__(self):
      self._iterator = iter(self._dataloader)
      return self
    
    def __next__(self):
      if self._length is None:
        return next(self._iterator)

      if self._index < self._length:
        self._index += 1
        try:
          return next(self._iterator)
        except StopIteration:
          raise RuntimeError('dataset length per rank out of real splitting length')

      self._index %= self._length

      while True:
          next(self._iterator)
      

