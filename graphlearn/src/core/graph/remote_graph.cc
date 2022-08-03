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

#include "core/graph/graph.h"

#include "common/base/errors.h"
#include "core/graph/storage/graph_storage.h"
#include "include/client.h"

namespace graphlearn {

class RemoteGraph : public Graph {
public:
  RemoteGraph(const std::string& type,
              const std::string& view_type,
              const std::string& use_attrs) {
    local_ = CreateLocalGraph(type, view_type, use_attrs);
  }

  virtual ~RemoteGraph() {
    delete local_;
  }

  Status Build(const IndexOption& option) override {
    return local_->Build(option);
  }

  io::GraphStorage* GetLocalStorage() override {
    return local_->GetLocalStorage();
  }

  Status UpdateEdges(const UpdateEdgesRequest* req,
                     UpdateEdgesResponse* res) override {
    return local_->UpdateEdges(req, res);
  }

  Status UpdateEdges(int32_t remote_id,
                     const UpdateEdgesRequest* req,
                     UpdateEdgesResponse* res) override {
    std::unique_ptr<Client> client(NewRpcClient(remote_id));
    return client->UpdateEdges(req, res);
  }

  Status LookupEdges(const LookupEdgesRequest* req,
                     LookupEdgesResponse* res) override {
    return local_->LookupEdges(req, res);
  }

  Status LookupEdges(int32_t remote_id,
                     const LookupEdgesRequest* req,
                     LookupEdgesResponse* res) override {
    std::unique_ptr<Client> client(NewRpcClient(remote_id));
    return client->LookupEdges(req, res);
  }

private:
  Graph* local_;
};

Graph* CreateRemoteGraph(const std::string& type,
                         const std::string& view_type,
                         const std::string& use_attrs) {
  return new RemoteGraph(type, view_type, use_attrs);
}

}  // namespace graphlearn
