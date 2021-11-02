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

class Server(object):
  """ Class to manage a GraphLearn server, including start, init and stop.
  """

  def __init__(self,
               server_id,
               server_count,
               server_host = "0.0.0.0:0",
               tracker = "",
               **kwargs):  # pylint: disable=unused-argument
    """ Create GraphLearn server instance with given cluster env info.

    Args:
      server_id: An int, server index in the cluster, starts from 0.
      server_count: An int, server instance number in the cluster.
      server_host: Use a given ip:port to start the server, which will be used
        for communication. By default, the server will choose an available
        address automatically. In this way, tracker is needed for address
        discovering.
      tracker: A string, the tracker path, where all server addresses will be
        registered to.
    """
    self._server_id = server_id
    self._server_count = server_count
    self._tracker = tracker
    self._server = pywrap.server(server_id, server_count, server_host, tracker)

  @property
  def server_id(self):
    return self._server_id

  @property
  def server_count(self):
    return self._server_count

  @property
  def tracker(self):
    return self._tracker

  def start(self):
    """ Start the server. It will not return until all servers are started.
    """
    self._server.start()

  def init(self, edge_source=None, node_source=None):
    """ Initialize graph data. It will not return until all data is installed.
    """
    if not edge_source:
      edge_source = []
    if not node_source:
      node_source = []
    self._server.init(edge_source, node_source)

  def stop(self):
    self._server.stop()

  def stop_sampling(self):
    self._server.stop_sampling()

  def get_stats(self):
    return self._server.get_stats()
