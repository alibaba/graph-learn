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
"""Abstract classes for EgoGraph and attributes encoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class BaseGraphEncoder(object):
  """Abstract class for Graph Encoder"""
  __metaclass__ = ABCMeta
  @abstractmethod
  def encode(self, ego_graph_tensor):
    """Encode ego_graph_tensor to embeddings.
    Args:
      ego_graph_tensor: an EgoGraphTensor instance
    Returns:
      embeddings
    """

    pass

class BaseFeatureEncoder(object):
  """Abstract class for Feature Encoder"""
  __metaclass__ = ABCMeta
  @abstractmethod
  def encode(self, input_attrs):
    """Encode input_attrs to embeddings.
    Args:
      input_attrs: A list of tensors recording input attributes, with
      format [continuous_attrs, categorical_attrs].
    Returns:
      embeddings.
    """
    pass
