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

import argparse
import json
import os


def install(dgs_host, json_file):
  url = dgs_host + "/admin/init"

  with open(json_file, 'r') as f:
    install_query_req = json.loads(f.read())
  content_data = json.dumps(install_query_req)

  res = os.system("curl -X POST -H \"Content-Type: text/plain\" -d \'" + content_data + "\' " + url)
  print(res)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Query Plan Installation Helper')
  parser.add_argument('--dgs-host', action="store", dest="dgs_host",
                      help="the host name of dynamic graph service")
  parser.add_argument('--install-query-json-file', action="store", dest="install_query_json_file",
                      help="the json file of install query request")
  args = parser.parse_args()

  if args.dgs_host is None:
    raise RuntimeError("The host of dynamic graph service must be specified!")

  if args.install_query_json_file is None:
    raise RuntimeError("The json file of install query request must be specified!")
  if not os.path.exists(args.install_query_json_file):
    raise RuntimeError("Missing install query request json file: {}!".format(args.install_query_json_file))

  install(args.dgs_host, args.install_query_json_file)
