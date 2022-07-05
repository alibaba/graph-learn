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

#include "include/server.h"

#include "include/config.h"
#include "service/server_impl.h"

namespace graphlearn {

Server::Server(ServerImpl* impl) : impl_(impl) {
}

Server::~Server() {
  delete impl_;
}

void Server::Start() {
  impl_->Start();
}

void Server::Init(const std::vector<io::EdgeSource>& edges,
                  const std::vector<io::NodeSource>& nodes) {
  impl_->Init(edges, nodes);
}

void Server::Stop() {
  impl_->Stop();
}

void Server::StopSampling() {
  impl_->StopSampling();
}

const Counts& Server::GetStats() const {
  return impl_->GetStats();
}


Server* NewServer(int32_t server_id,
                  int32_t server_count,
                  const std::string& server_host,
                  const std::string& tracker) {
  ServerImpl* impl = GLOBAL_FLAG(EnableActor) ?
    NewActorServerImpl(server_id, server_count, server_host, tracker) :
    NewDefaultServerImpl(server_id, server_count, server_host, tracker);
  return new Server(impl);
}

}  // namespace graphlearn