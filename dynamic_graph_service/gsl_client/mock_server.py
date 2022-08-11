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
import urllib.parse


class MockHttpHandler(BaseHTTPRequestHandler):
  grpc_server = None

  def do_POST(self):
    content_length = int(self.headers['Content-Length'])
    body = self.rfile.read(content_length)
    if (self.path == "/admin/init/"):
      msg = body.decode("utf-8")
      response = BytesIO()
      try:
        json.loads(msg)
        self.send_response(200)
        self.end_headers()
        response.write(b'0')
        self.wfile.write(response.getvalue())
        return
      except:
        logging.error("Parse query failed.")
    self.send_response(400)
    self.end_headers()
    response.write(b'HTTP RESPONSE: Query installed FAILED with Wrong JSON format for Query Plan.\n')
    self.wfile.write(response.getvalue())

  def do_GET(self):
    response = BytesIO()
    path = self.path.split('?')
    if (path[0] == "/infer"):
      if len(path) > 1:
        params = path[1]
        o = urllib.parse.parse_qs(params)
        if o.get("qid") and o.get("vid"):
          self.send_response(200)
        self.end_headers()
        response.write(b'HTTP RESPONSE: Run Query SUCCEED.\n')
        self.wfile.write(response.getvalue())
        return
    elif (path[0] == "/admin/schema/"):
      self.send_response(200)
      self.end_headers()
      with open("../conf/ut/schema.ut.json", "rb") as f:
        try:
          self.wfile.write(f.read())
          return
        except:
          print("Parse schema file failed....")
    self.send_response(400)
    response.write(b'HTTP RESPONSE: GET FAILED.\n')
    self.wfile.write(response.getvalue())

class MockHttpService(object):
  def __init__(self, port):
    self._port = port
    self._server = HTTPServer(('', self._port), MockHttpHandler)

  def start(self):
    self._server.serve_forever()

  def stop(self):
    self._server.server_close()

if __name__ == "__main__":
  service = MockHttpService(8088)
  try:
    service.start()
  except KeyboardInterrupt:
    pass
  service.stop()
