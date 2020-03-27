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

#include "graphlearn/core/graph/noder.h"

#include "graphlearn/common/base/errors.h"
#include "graphlearn/core/graph/storage/node_storage.h"

namespace graphlearn {

class LocalNoder : public Noder {
public:
  LocalNoder() {
    storage_ = io::NewMemoryNodeStorage();
  }

  virtual ~LocalNoder() {
    delete storage_;
  }

  void Build() override {
    storage_->Build();
  }

  io::NodeStorage* GetLocalStorage() override {
    return storage_;
  }

  Status UpdateNodes(const UpdateNodesRequest* req,
                     UpdateNodesResponse* res) override {
    storage_->Lock();

    storage_->SetSideInfo(req->GetSideInfo());

    io::NodeValue value;
    UpdateNodesRequest* request = const_cast<UpdateNodesRequest*>(req);
    while (request->Next(&value)) {
      storage_->Add(&value);
    }

    storage_->Unlock();
    return Status::OK();
  }

  Status UpdateNodes(int32_t remote_id,
                     const UpdateNodesRequest* req,
                     UpdateNodesResponse* res) override {
    return error::Unimplemented("Remote UpdateNodes not implemented");
  }

  Status LookupNodes(const LookupNodesRequest* req,
                     LookupNodesResponse* res) override {
    int64_t node_id = 0;
    LookupNodesRequest* request = const_cast<LookupNodesRequest*>(req);
    res->SetSideInfo(storage_->GetSideInfo(), req->Size());
    while (request->Next(&node_id)) {
      res->AppendWeight(storage_->GetWeight(node_id));
      res->AppendLabel(storage_->GetLabel(node_id));
      res->AppendAttribute(storage_->GetAttribute(node_id));
    }
    return Status::OK();
  }

  Status LookupNodes(int32_t remote_id,
                     const LookupNodesRequest* req,
                     LookupNodesResponse* res) override {
    return error::Unimplemented("Remote LookupNodes not implemented");
  }

private:
  io::NodeStorage* storage_;
};

Noder* CreateLocalNoder() {
  return new LocalNoder();
}

}  // namespace graphlearn
