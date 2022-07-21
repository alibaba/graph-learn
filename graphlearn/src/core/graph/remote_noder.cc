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

#include "core/graph/noder.h"

#include "common/base/errors.h"
#include "core/graph/storage/node_storage.h"
#include "include/client.h"

namespace graphlearn {

class RemoteNoder : public Noder {
public:
  RemoteNoder(const std::string& type,
              const std::string& view_type,
              const std::string &use_attrs) {
    local_ = CreateLocalNoder(type, view_type, use_attrs);
  }

  virtual ~RemoteNoder() {
    delete local_;
  }

  Status Build(const IndexOption& option) override {
    return local_->Build(option);
  }

  io::NodeStorage* GetLocalStorage() override {
    return local_->GetLocalStorage();
  }

  Status UpdateNodes(const UpdateNodesRequest* req,
                     UpdateNodesResponse* res) override {
    return local_->UpdateNodes(req, res);
  }

  Status UpdateNodes(int32_t remote_id,
                     const UpdateNodesRequest* req,
                     UpdateNodesResponse* res) override {
    std::unique_ptr<Client> client(NewRpcClient(remote_id));
    return client->UpdateNodes(req, res);
  }

  Status LookupNodes(const LookupNodesRequest* req,
                     LookupNodesResponse* res) override {
    return local_->LookupNodes(req, res);
  }

  Status LookupNodes(int32_t remote_id,
                     const LookupNodesRequest* req,
                     LookupNodesResponse* res) override {
    std::unique_ptr<Client> client(NewRpcClient(remote_id));
    return client->LookupNodes(req, res);
  }

private:
  Noder* local_;
};

Noder* CreateRemoteNoder(const std::string& type,
                         const std::string& view_type,
                         const std::string &use_attrs) {
  return new RemoteNoder(type, view_type, use_attrs);
}

}  // namespace graphlearn
