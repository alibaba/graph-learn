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

class Config(object):
  def __init__(self):
    """Set and get configurations for tf models.

    Configurations:
      training (bool): Whether in training mode or not. Defaults to True.
      emb_max_partitions (int): The `max_partitions` for embedding variables
        partitioned by `min_max_variable_partitioner`. Specially,
        `EmbeddingVariable` uses `fixed_size_partitioner`.
        Defaults to None means no partition.
      emb_min_slice_size (int): The `min_slice_size` for embedding variables
        partitioned by `min_max_variable_partitioner`. Defaults to 128K.
      emb_live_steps (int): Global steps to live for inactive keys in embedding
        variables. Defaults to None.
    """
    self.training = True
    self.partitioner = 'min_max'
    self.emb_max_partitions = None
    self.emb_min_slice_size = 128 * 1024
    self.emb_live_steps = None

conf = Config()
