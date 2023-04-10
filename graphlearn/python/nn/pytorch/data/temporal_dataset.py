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

from graphlearn.python.nn.pytorch.data.dataset import Dataset

class TemporalDataset(Dataset):
 def __init__(self, query, window=10, event_name="event"):
  super().__init__(query, window=window)
  def _induce_func(data_dict):
    if data_dict.get(event_name) is None:
      raise ValueError("Event name {} not exist.".format(event_name))
    events = data_dict[event_name]
    res = TemporalData(
      torch.from_numpy(events.ids),
      torch.from_numpy(events.dst_ids),
      torch.from_numpy(events.timestamps),
      torch.from_numpy(events.weights)
    )
    return res
  self._induce_func = _induce_func
