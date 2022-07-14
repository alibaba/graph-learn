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
  schema_file = ""
  meta = None

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
        response.write(b'HTTP RESPONSE: INSTALL SUCCESSFUL.\n')
      except:
        qid[0] = None
        logging.info("Register query failed, with wrong query plan.")
        response.write(b'HTTP RESPONSE: INSTALL FAILED.\n')
      self.wfile.write(response.getvalue())
    elif self.path.startswith("/admin/checkpoint"):
      self.send_response(200)
      self.end_headers()
      response = BytesIO()
      sampling_res, serving_res = self.__class__.grpc_server.start_checkpointing()
      response.write(bytes("CREATING SAMPLING WORKER CHECKPOINT {}.\n"
                           .format("SUCCESSFUL" if sampling_res else "FAILED")))
      response.write(bytes("CREATING SERVING WORKER CHECKPOINT {}.\n"
                           .format("SUCCESSFUL" if serving_res else "FAILED")))
      self.wfile.write(response.getvalue())

  def do_GET(self):
    response = BytesIO()
    if self.path.startswith("/admin/schema"):
      self.send_response(200)
      self.end_headers()
      with open(self.schema_file, "rb") as f:
        try:
          self.wfile.write(f.read())
          return
        except:
          logging.error("Failed to read schema file ...")
    self.send_response(400)
    response.write(b'HTTP RESPONSE: GET FAILED.\n')
    self.wfile.write(response.getvalue())


class CoordinatorHttpService(object):
  def __init__(self, configs, grpc_server, meta, port=8080):
    CoordinatorHttpHandler.grpc_server = grpc_server
    CoordinatorHttpHandler.meta = meta
    CoordinatorHttpHandler.schema_file = configs.get("schema_file")
    self._port = port
    self._server = HTTPServer(('', self._port), CoordinatorHttpHandler)

  def start(self):
    logging.info("Http Server for Coordinator running on port {}.\n".format(self._port))
    self._server.serve_forever()

  def stop(self):
    self._server.server_close()
    logging.info('Stopping Http Server...\n')
