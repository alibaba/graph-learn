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

#include "graphlearn/include/client.h"

#include <mutex>  // NOLINT [build/c++11]
#include <vector>
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/include/config.h"
#include "graphlearn/service/client_impl.h"

namespace graphlearn {

Client::Client(ClientImpl* impl, bool own)
    : impl_(impl), own_(own) {
}

Client::~Client() {
  if (own_) {
    delete impl_;
  }
}

#define DEFINE_METHOD(Name)                       \
Status Client::Name(const Name##Request* request, \
                    Name##Response* response) {   \
  return impl_->RunOp(request, response);         \
}

DEFINE_METHOD(UpdateEdges);
DEFINE_METHOD(UpdateNodes);
DEFINE_METHOD(GetEdges);
DEFINE_METHOD(LookupEdges);
DEFINE_METHOD(GetNodes);
DEFINE_METHOD(LookupNodes);
DEFINE_METHOD(GetTopology);
DEFINE_METHOD(Sampling);
DEFINE_METHOD(Aggregating);

#undef DEFINE_METHOD

Status Client::RunOp(const OpRequest* request, OpResponse* response) {
  return impl_->RunOp(request, response);
}

Status Client::Stop() {
  return impl_->Stop();
}

Client* NewInMemoryClient() {
  return new Client(NewInMemoryClientImpl());
}

class ClientManager {
public:
  ClientManager() {
    clients_.resize(GLOBAL_FLAG(ServerCount), nullptr);
  }

  ~ClientManager() {
    for (auto& item : clients_) {
      delete item;
    }
  }

  ClientImpl* LookupOrCreate(int32_t server_id, bool server_own) {
    if (server_id >= GLOBAL_FLAG(ServerCount)) {
      LOG(FATAL) << "Unexpected server id: " << server_id;
      return nullptr;
    }

    ScopedLocker<std::mutex> _(&mtx_);
    if (clients_[server_id] == nullptr) {
      ClientImpl* c = NewRpcClientImpl(server_id, server_own);
      clients_[server_id] = c;
    }
    return clients_[server_id];
  }

private:
  std::mutex mtx_;
  std::vector<ClientImpl*> clients_;
};

Client* NewRpcClient(int32_t server_id, bool server_own) {
  static ClientManager manager;
  if (server_id < 0) {
    // auto select
    return new Client(NewRpcClientImpl(server_id, server_own), true);
  } else {
    return new Client(manager.LookupOrCreate(server_id, server_own), false);
  }
}

}  // namespace graphlearn
