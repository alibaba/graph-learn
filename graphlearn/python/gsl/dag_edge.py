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

from graphlearn import pywrap_graphlearn as pywrap


class DagEdge(object):
  def __init__(self, eid):
    self._dag_edge_def = pywrap.new_dag_edge()
    self._id = eid
    self._src = None  # Source DagNode.
    self._dst = None  # Dst DagNode.
    self._src_output = None  # Broadcast which field of Source DagNode.
    self._dst_input = None  # Fill in which field of Dst DagNode.

    pywrap.set_dag_edge_id(self._dag_edge_def, eid)

  @property
  def src(self):
    # Source DagNode.
    return self._src

  @src.setter
  def src(self, src):
    self._src = src

  @property
  def dst(self):
    # Dst DagNode.
    return self._dst

  @dst.setter
  def dst(self, dst):
    self._dst = dst

  @property
  def src_output(self):
    return self._src_output

  @src_output.setter
  def src_output(self, src_output):
    self._src_output = src_output
    pywrap.set_dag_edge_src_output(self._dag_edge_def, src_output)

  @property
  def dst_input(self):
    return self._dst_input

  @dst_input.setter
  def dst_input(self, dst_input):
    self._dst_input = dst_input
    pywrap.set_dag_edge_dst_input(self._dag_edge_def, dst_input)

  @property
  def dag_edge_def(self):
    return self._dag_edge_def

  @property
  def edge_id(self):
    return self._id


dag_edges = {}


def get_dag_edge(eid):
  global dag_edges
  if not dag_edges.get(eid):
    dag_edges[eid] = DagEdge(eid)
  return dag_edges[eid]


eid = 0


def get_eid():
  global eid
  eid += 1
  return eid
