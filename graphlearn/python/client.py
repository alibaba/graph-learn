# Copyright 2020-2022 Alibaba Group Holding Limited. All Rights Reserved.
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
""" GraphLearn Client.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graphlearn import pywrap_graphlearn as pywrap


class Client(object):
  """ A :code:`Client` class manages a set of remote server to connect
      and do a round-robin sample between those GraphLearn clients.
  """

  def __init__(self,
               client_id = 0,
               in_memory = True,
               **kwargs):  # pylint: disable=unused-argument
    """ Create GraphLearn server instance with given cluster env info.
    Args:
      client_id: An int, client index, starts from 0.
      in_memory: An bool, is in memory client or rpc client.
    """
    self._client_id = client_id
    self._in_memory = in_memory
    self._client_cache = []
    self._client_own = kwargs.get("client_own", True)
    if in_memory:
      self._own_servers = None
      self._client_cache.append(pywrap.in_memory_client())
      self._current_index = 0
    else:
      server_id = kwargs.get("server_id", -1)
      # auto select
      client = pywrap.rpc_client(server_id, self._client_own)
      self._own_servers = client.get_own_servers()
      self._client_cache = [None] * len(self._own_servers)
      self._client_cache[0] = pywrap.rpc_client(self._own_servers[0], self._client_own)
      self._current_index = 0

  def __len__(self):
    """ The capacity of the clients inside the client wrapper.
    """
    return len(self._client_cache)

  def is_in_memory(self):
    return self._in_memory

  @property
  def current_client(self):
    return self._client_cache[self._current_index]

  def connect_to_next_server(self):
    if self._in_memory:
      return False

    if len(self._client_cache) == 1:  # the client has only one own server
      return False

    # move to the next server
    next_index = (self._current_index + 1) % len(self._own_servers)
    if next_index < self._current_index:
      # one epoch finish, start next epoch
      self._current_index = next_index
      return False
    else:
      # one server finish, start to connect next server
      if self._client_cache[next_index] is None:
        self._client_cache[next_index] = pywrap.rpc_client(
          self._own_servers[next_index], self._client_own
        )
      self._current_index = next_index
      return True

  def stop(self):
    for client in self._client_cache:
      if client is not None:
        client.stop()
    self._client_cache.clear()

  def get_nodes(self, request, response):
    return self.current_client.get_nodes(request, response)

  def get_edges(self, request, response):
    return self.current_client.get_edges(request, response)

  def lookup_nodes(self, request, response):
    return self.current_client.lookup_nodes(request, response)

  def lookup_edges(self, request, response):
    return self.current_client.lookup_edges(request, response)

  def sample_neighbor(self, request, response):
    return self.current_client.sample_neighbor(request, response)

  def agg_nodes(self, request, response):
    return self.current_client.agg_nodes(request, response)

  def sample_subgraph(self, request, response):
    return self.current_client.sample_subgraph(request, response)

  def get_stats(self, request, response):
    return self.current_client.get_stats(request, response)

  def get_degree(self, request, response):
    return self.current_client.get_degree(request, response)

  def run_op(self, request, response):
    return self.current_client.run_op(request, response)

  def run_dag(self, dag_def, copy=False):
    return self.current_client.run_dag(dag_def, copy)

  def get_dag_values(self, request, response):
    return self.current_client.get_dag_values(request, response)

  def cond_neg_sample(self, request, response):
    return self.current_client.cond_neg_sample(request, response)
