/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef GRAPHLEARN_INCLUDE_SERVER_H_
#define GRAPHLEARN_INCLUDE_SERVER_H_

#include <cstdint>
#include <string>
#include <vector>
#include "graphlearn/include/data_source.h"
#include "graphlearn/include/status.h"

namespace graphlearn {

class ServerImpl;

class Server {
public:
  ~Server();

  void Start();

  void Init(const std::vector<io::EdgeSource>& edges,
            const std::vector<io::NodeSource>& nodes);

  void Stop();

private:
  explicit Server(ServerImpl* impl);

private:
  ServerImpl* impl_;

  friend Server* NewServer(int32_t server_id,
                           int32_t server_count,
                           const std::string& tracker);
};

Server* NewServer(int32_t server_id,
                  int32_t server_count,
                  const std::string& tracker);

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_SERVER_H_
