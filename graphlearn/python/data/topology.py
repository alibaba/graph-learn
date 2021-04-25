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

import warnings


class Topology(object):
  def __init__(self):
    self._topology = {}

  def add(self, edge_type, src_type, dst_type):
    if self._topology.get(edge_type):
      raise ValueError("edge_type {} has existed.".format(edge_type))
    self._topology[edge_type] = EdgeInfo(src_type, dst_type)

  def get_edge_info(self, edge_type):
    edge_info = self._topology.get(edge_type)
    if not edge_info:
      raise ValueError("edge_type {} not exists in the graph."
                       .format(edge_type))
    return edge_info

  def get_src_type(self, edge_type):
    edge_info = self._topology.get(edge_type)
    if not edge_info:
      raise ValueError("edge_type {} not exists in the graph."
                       .format(edge_type))
    return edge_info.src_type

  def get_dst_type(self, edge_type):
    edge_info = self._topology.get(edge_type)
    if not edge_info:
      raise ValueError("edge_type {} not exists in the graph."
                       .format(edge_type))
    return edge_info.dst_type

  def is_exist(self, edge_type):
    if self._topology.get(edge_type):
      return True
    return False

  def print_all(self):
    for k, v in self._topology.items():
      print("edge_type:{}, src_type:{}, dst_type:{}\n"
            .format(k, v.src_type, v.dst_type))

  def print_one(self, edge_type):
    edge_info = self._topology.get(edge_type)
    if not edge_info:
      warnings.warn("edge_type {} not exists in the graph."
                    .format(edge_type))
    print("edge_type:{}, src_type:{}, dst_type:{}\n"
          .format(edge_type, edge_info.src_type, edge_info.dst_type))


class EdgeInfo(object):
  def __init__(self, src_type, dst_type):
    self._src_type = src_type
    self._dst_type = dst_type

  @property
  def src_type(self):
    return self._src_type

  @property
  def dst_type(self):
    return self._dst_type
