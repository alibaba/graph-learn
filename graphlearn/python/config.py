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
""" Functions to set global configurations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.utils import deprecated


def set_default_neighbor_id(nbr_id):
  pywrap.set_default_neighbor_id(nbr_id)


def set_tracker_mode(mode):
  pywrap.set_tracker_mode(mode)


def set_padding_mode(mode):
  pywrap.set_padding_mode(mode)


def set_storage_mode(mode):
  pywrap.set_storage_mode(mode)


def set_default_int_attribute(value=0):
  """ Set default global int attribute.
  """
  pywrap.set_default_int_attr(int(value))


def set_default_float_attribute(value=0.0):
  """ Set default global float attribute.
  """
  pywrap.set_default_float_attr(float(value))


def set_default_string_attribute(value=''):
  """ Set default global string attribute.
  """
  pywrap.set_default_string_attr(str(value))


def set_timeout(time_in_second):
  pywrap.set_timeout(time_in_second)


def set_inmemory_queuesize(size):
  pywrap.set_inmemory_queuesize(size)


def set_inner_threadnum(num):
  pywrap.set_inner_threadnum(num)


def set_inter_threadnum(num):
  pywrap.set_inter_threadnum(num)


def set_intra_threadnum(num):
  pywrap.set_intra_threadnum(num)


def set_datainit_batchsize(size):
  pywrap.set_datainit_batchsize(size)


def set_shuffle_buffer_size(size):
  pywrap.set_shuffle_buffer_size(size)


def set_rpc_message_max_size(size):
  pywrap.set_rpc_message_max_size(size)


def set_vineyard_graph_id(graph_id):
  pywrap.set_vineyard_graph_id(graph_id)


def set_vineyard_ipc_socket(ipc_socket):
  pywrap.set_vineyard_ipc_socket(ipc_socket)


