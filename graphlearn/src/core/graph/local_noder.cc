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
#include "common/base/log.h"
#ifdef OPEN_KNN
#include "contrib/knn/builder.h"
#endif
#include "core/graph/storage/node_storage.h"
#include "core/graph/storage_creator.h"
#include "include/config.h"

namespace graphlearn {

class LocalNoder : public Noder {
public:
  LocalNoder(const std::string& type,
             const std::string& view_type,
             const std::string& use_attrs) {
    storage_ = CreateNodeStorage(type, view_type, use_attrs);
  }

  virtual ~LocalNoder() {
    delete storage_;
  }

  Status Build(const IndexOption& option) override {
    if (option.name == "sort") {
      storage_->Build();
    } else if (option.name == "knn") {
#ifdef OPEN_KNN
      if (!BuildKnnIndex(storage_, option)) {
        LOG(ERROR) << "Invalid node type or index type for building KNN index:"
                   << storage_->GetSideInfo()->type;
        return error::InvalidArgument("Invalid node type or index type.");
      }
#endif
    } else {
      USER_LOG("Unsupported node index type:" + option.name);
      LOG(WARNING) << "Unsupported node index type:" << option.name;
    }
    return Status();
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
      res->AppendTimestamp(storage_->GetTimestamp(node_id));
      res->AppendAttribute(storage_->GetAttribute(node_id).get());
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

Noder* CreateLocalNoder(const std::string& type,
                        const std::string& view_type,
                        const std::string& use_attrs) {
  return new LocalNoder(type, view_type, use_attrs);
}

}  // namespace graphlearn
