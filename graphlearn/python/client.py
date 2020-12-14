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
""" GraphLearn Client.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graphlearn import pywrap_graphlearn as pywrap


class Client(object):
  """ A class managing a GraphLearn client.
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
    if in_memory:
      self._client_cache.append(pywrap.in_memory_client())
      self._cur_index = 0
    else:
      server_id = kwargs.get("server_id", -1)
      server_own = kwargs.get("server_own", False)
      client_own = kwargs.get("client_own", True)
      # auto select
      client = pywrap.rpc_client(server_id, server_own, client_own)
      self._own_servers = client.get_own_servers()
      self._client_cache = [None] * len(self._own_servers)
      self._client_cache[0] = client
      self._cur_index = 0

  @property
  def own_servers(self):
    return self._own_servers

  def connect_to_next_server(self):
    if self._in_memory:
      return False

    next_index = (self._cur_index + 1) % len(self._own_servers)
    if next_index == self._cur_index:
      return False   # the client has only one own server
    elif next_index < self._cur_index:
      # one epoch finish, start next epoch
      self._cur_index = next_index
      return False
    else:
      # one server finish, start to connect next server
      if self._client_cache[next_index] is None:
        self._client_cache[next_index] = pywrap.rpc_client(
          self._own_servers[next_index], True, True
        )
      self._cur_index = next_index
      return True

  def stop(self):
    for client in self._client_cache:
      if client is not None:
        client.stop()
    self._client_cache.clear()

  def lookup_nodes(self, req, res):
    return self._client_cache[self._cur_index].lookup_nodes(req, res)

  def lookup_edges(self, req, res):
    return self._client_cache[self._cur_index].lookup_edges(req, res)

  def get_nodes(self, req, res):
    return self._client_cache[self._cur_index].get_nodes(req, res)

  def get_edges(self, req, res):
    return self._client_cache[self._cur_index].get_edges(req, res)

  def agg_nodes(self, req, res):
    return self._client_cache[self._cur_index].agg_nodes(req, res)

  def sample_neighbor(self, req, res):
    return self._client_cache[self._cur_index].sample_neighbor(req, res)

  def is_in_memory(self):
    return self._in_memory
