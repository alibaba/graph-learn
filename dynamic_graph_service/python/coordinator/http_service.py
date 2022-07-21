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
    content_length = int(self.headers['Content-Length'])
    body = self.rfile.read(content_length)
    if self.path.startswith("/admin/init"):
      self.send_response(200)
      self.end_headers()
      response = BytesIO()
      msg = body.decode("utf-8")
      qid = [None]
      try:
        json.loads(msg)
        # msg = msg.replace("\"", "")
        # msg = msg.replace("\\\\", "\"")
        res = self.__class__.meta.register(qid, msg)
        if res:
          self.__class__.grpc_server.init_query(msg)
        response.write(b'Install Successfully.\n')
      except:
        qid[0] = None
        logging.info("Register query failed, with wrong query plan.")
        response.write(b'Install Failed.\n')
      self.wfile.write(response.getvalue())
    elif self.path.startswith("/admin/checkpoint"):
      self.send_response(200)
      self.end_headers()
      response = BytesIO()
      sampling_res, serving_res = self.__class__.grpc_server.start_checkpointing()
      response.write(bytes("Creating Sampling Worker Checkpoint {}.\n"
                           .format("Successfully" if sampling_res else "Failed")))
      response.write(bytes("Creating Serving Worker Checkpoint {}.\n"
                           .format("Successfully" if serving_res else "Failed")))
      self.wfile.write(response.getvalue())
    elif self.path.startswith("/admin/barrier"):
      self.send_response(200)
      self.end_headers()
      barrier_name = None
      dl_count = None
      dl_id = None
      params = body.decode("utf-8").split("&")
      for param in params:
        kv = param.split("=")
        if kv[0] == "name":
          barrier_name = kv[1]
        elif kv[0] == "dl_count":
          dl_count = int(kv[1])
        elif kv[0] == "dl_id":
          dl_id = int(kv[1])
      if (barrier_name is None) or (dl_count is None) or (dl_id is None):
        self.wfile.write(b'Wrong Params.\n')
      else:
        self.barrier_monitor.set_barrier_from_dataloader(barrier_name, dl_count, dl_id)
        self.wfile.write(b'Done')
    else:
      self.send_response(400)
      self.end_headers()
      self.wfile.write(b'Unsupported POST Request.\n')

  def do_GET(self):
    if self.path.startswith("/admin/schema"):
      self.send_response(200)
      self.end_headers()
      self.wfile.write(bytes(self.schema))
    elif self.path.startswith("/admin/dataloader-init-info"):
      dl_init_info = {
        "downstream": self.dl_ds_info,
        "schema": self.schema_json
      }
      dl_init_json_str = json.dumps(dl_init_info)
      self.send_response(200)
      self.end_headers()
      self.wfile.write(bytes(dl_init_json_str, "UTF-8"))
    elif self.path.startswith("/admin/barrier-status"):
      self.send_response(200)
      self.end_headers()
      param = self.path.split("?")[1]
      kv = param.split("=")
      if kv[0] == "name":
        barrier_name = kv[1]
        if self.barrier_monitor.check_ready(barrier_name):
          self.wfile.write(b'Ready')
        else:
          self.wfile.write(b'Not Ready')
      else:
        self.wfile.write(b'Missing Barrier Name.\n')
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
