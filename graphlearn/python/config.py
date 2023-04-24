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

import functools
from graphlearn import pywrap_graphlearn as pywrap


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

def set_default_weight(value=0.0):
  pywrap.set_default_weight(value)

def set_default_label(value=-1):
  pywrap.set_default_label(value)

def set_default_timestamp(value=-1):
  pywrap.set_default_timestamp(value)

def set_timeout(time_in_second):
  pywrap.set_timeout(time_in_second)

def set_retry_times(retry_times):
  pywrap.set_retry_times(retry_times)

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

def set_knn_metric(metric):
  '''
  Args:
    metric: 0 is l2 distance, 1 is inner product.
  '''
  pywrap.set_knn_metric(metric)

def set_dataset_capacity(size):
  assert 0 < size < 128, "Dataset capacity should be > 0 and < 128."
  pywrap.set_dataset_capacity(size)

def set_tape_capacity(size):
  assert 0 < size < 128, "Tape capacity should be > 0 and < 128."
  pywrap.set_tape_capacity(size)

def set_ignore_invalid(value):
  pywrap.set_ignore_invalid(value)

def set_default_full_nbr_num(num):
  pywrap.set_default_full_nbr_num(num)

def set_local_node_cache_capacity(count):
  assert 0 <= count, "local node cache count should be >= 0."
  pywrap.set_local_node_cache_capacity(count)

def enable_actor():
  pywrap.set_enable_actor(1)

def set_actor_local_shard_count(count):
  assert isinstance(count, int) and count > 0
  pywrap.set_actor_local_shard_count(count)

def set_sampler_retry_times(times):
  pywrap.set_sampler_retry_times(times)

def set_field_delimiter(delimiter="\t"):
  pywrap.set_field_delimiter(delimiter)

def set_vineyard_graph_id(graph_id):
  pywrap.set_vineyard_graph_id(graph_id)

def set_vineyard_ipc_socket(ipc_socket):
  pywrap.set_vineyard_ipc_socket(ipc_socket)
