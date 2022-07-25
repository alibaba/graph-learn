# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
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

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import urlparse, parse_qsl


class CoordinatorHttpHandler(BaseHTTPRequestHandler):
  grpc_server = None
  barrier_monitor = None
  meta = None
  schema = ""
  schema_json = {}
  dl_ds_info = {}

  def do_POST(self):
    if self.__class__.grpc_server is None:
      logging.error("Grpc server is not set successfully.\n")
    url_parsed = urlparse(self.path)
    content_length = int(self.headers['Content-Length'])
    content = self.rfile.read(content_length)
    if url_parsed.path is "/admin/init":
      qid = [None]
      json_str = content.decode("utf-8")
      try:
        # TODO(@goldenleaves): check schema json.
        json.loads(json_str)
        res = self.__class__.meta.register(qid, json_str)
        if res:
          self.__class__.grpc_server.init_query(json_str)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Install Successfully.\n')
      except:
        self.send_response(400)
        self.end_headers()
        self.wfile.write(b'Invalid Query Plan\n')
    elif url_parsed.path is "/admin/checkpoint":
      response = BytesIO()
      sampling_res, serving_res = self.__class__.grpc_server.start_checkpointing()
      response.write(bytes("Creating Sampling Worker Checkpoint {}.\n"
                           .format("Successfully" if sampling_res else "Failed")))
      response.write(bytes("Creating Serving Worker Checkpoint {}.\n"
                           .format("Successfully" if serving_res else "Failed")))
      self.send_response(200)
      self.end_headers()
      self.wfile.write(response.getvalue())
    elif url_parsed.path is "/admin/barrier/set":
      params = dict(parse_qsl(content.decode("utf-8")))
      barrier_name = params.get("name")
      dl_count = params.get("count")
      dl_id = params.get("id")
      res_code = 400
      response = BytesIO()
      if (barrier_name is None) or (dl_count is None) or (dl_id is None):
        response.write(b'Wrong Parameters.\n')
      elif self.barrier_monitor.check_existed(barrier_name):
        response.write(bytes("Barrier {} Already Set.\n".format(barrier_name)))
      else:
        res_code = 200
        self.barrier_monitor.set_barrier_from_dataloader(barrier_name, int(dl_count), int(dl_id))
        response.write(b'Done')
      self.send_response(res_code)
      self.end_headers()
      self.wfile.write(response.getvalue())
    else:
      self.send_response(400)
      self.end_headers()
      self.wfile.write(b'Unsupported POST Request.\n')

  def do_GET(self):
    url_parsed = urlparse(self.path)
    if url_parsed.path is "/admin/schema":
      self.send_response(200)
      self.end_headers()
      self.wfile.write(bytes(self.schema))
    elif url_parsed.path is "/admin/init-info/dataloader":
      dl_init_info = {
        "downstream": self.dl_ds_info,
        "schema": self.schema_json
      }
      dl_init_json_str = json.dumps(dl_init_info)
      self.send_response(200)
      self.end_headers()
      self.wfile.write(bytes(dl_init_json_str, "UTF-8"))
    elif url_parsed.path is "/admin/barrier/status":
      params = dict(parse_qsl(url_parsed.query))
      barrier_name = params.get("name")
      res_code = 400
      response = BytesIO()
      if barrier_name is None:
        response.write(b'Wrong Parameters: Missing Barrier Name.\n')
      else:
        res_code = 200
        status = self.barrier_monitor.check_status(barrier_name)
        response.write(bytes(status))
      self.send_response(res_code)
      self.end_headers()
      self.wfile.write(response.getvalue())
    else:
      self.send_response(400)
      self.end_headers()
      self.wfile.write(b'Unsupported GET Request.\n')


class CoordinatorHttpService(object):
  def __init__(self, port, grpc_server, barrier_monitor, meta, schema, dl_ds_info):
    CoordinatorHttpHandler.grpc_server = grpc_server
    CoordinatorHttpHandler.barrier_monitor = barrier_monitor
    CoordinatorHttpHandler.meta = meta
    CoordinatorHttpHandler.schema = schema
    CoordinatorHttpHandler.schema_json = json.loads(schema)
    CoordinatorHttpHandler.dl_ds_info = dl_ds_info
    self._port = port
    self._server = HTTPServer(('', self._port), CoordinatorHttpHandler)

  def start(self):
    logging.info("Http Server for Coordinator running on port {}.\n".format(self._port))
    CoordinatorHttpHandler.barrier_monitor.start()
    self._server.serve_forever()

  def stop(self):
    self._server.server_close()
    CoordinatorHttpHandler.barrier_monitor.stop()
    logging.info('Stopping Http Server...\n')
