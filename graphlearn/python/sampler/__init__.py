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
"""All the samplers, include node, edge, neighbor and negative.
"""
from graphlearn.python.sampler.node_sampler import RandomNodeSampler, \
  ByOrderNodeSampler, ShuffleNodeSampler
from graphlearn.python.sampler.edge_sampler import RandomEdgeSampler, \
  ByOrderEdgeSampler, ShuffleEdgeSampler
from graphlearn.python.sampler.neighbor_sampler import RandomNeighborSampler, \
  EdgeWeightNeighborSampler, TopkNeighborSampler, \
  InDegreeNeighborSampler, FullNeighborSampler, \
  RandomWithoutReplacementNeighborSampler
from graphlearn.python.sampler.negative_sampler import RandomNegativeSampler, \
  InDegreeNegativeSampler, NodeWeightNegativeSampler

__all__ = [
    "RandomNodeSampler",
    "ByOrderNodeSampler",
    "ShuffleNodeSampler",
    "RandomEdgeSampler",
    "ByOrderEdgeSampler",
    "ShuffleEdgeSampler",
    "RandomNeighborSampler",
    "RandomWithoutReplacementNeighborSampler",
    "EdgeWeightNeighborSampler",
    "InDegreeNeighborSampler",
    "FullNeighborSampler",
    "RandomNegativeSampler",
    "InDegreeNegativeSampler",
    "NodeWeightNegativeSampler"
]
