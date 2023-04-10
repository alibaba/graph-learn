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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import torch
except ImportError:
  pass
try:
  from torch_geometric.data.temporal import TemporalData
except ImportError:
  pass

from graphlearn.python.nn.pytorch.data.temporal_dataset import TemporalDataset

class TemporalDataLoader(torch.utils.data.DataLoader):
  def __init__(self, dataset, **kwargs):
      assert isinstance(dataset, TemporalDataset), 'TemporalDataLoader only accepts GraphLearn TemporalDataset'
      if "batch_size" in kwargs:
        del kwargs["batch_size"]
      if "collate_fn" in kwargs:
        del kwargs["collate_fn"]
      if 'shuffle' in kwargs:
        del kwargs['shuffle']
      super().__init__(dataset, None, shuffle=False, collate_fn=self, **kwargs)


  def __call__(self, data):
    return data
