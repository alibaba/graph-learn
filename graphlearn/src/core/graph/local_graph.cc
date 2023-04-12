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
#include "core/graph/storage_creator.h"
#include "include/config.h"

namespace graphlearn {

class LocalGraph : public Graph {
public:
  LocalGraph(const std::string& type,
             const std::string& view_type,
             const std::string& use_attrs) {
    storage_ = CreateGraphStorage(type, view_type, use_attrs);
  }

  virtual ~LocalGraph() {
    delete storage_;
  }

  Status Build(const IndexOption& option) override {
    if (option.name == "sort") {
      storage_->Build();
    } else {
      // Nothing to do
    }
    return Status();
  }

  io::GraphStorage* GetLocalStorage() override {
    return storage_;
  }

  Status UpdateEdges(const UpdateEdgesRequest* req,
                     UpdateEdgesResponse* res) override {
    storage_->Lock();

    storage_->SetSideInfo(req->GetSideInfo());

    io::EdgeValue value;
    UpdateEdgesRequest* request = const_cast<UpdateEdgesRequest*>(req);
    while (request->Next(&value)) {
      storage_->Add(&value);
    }

    storage_->Unlock();
    return Status::OK();
  }

  Status UpdateEdges(int32_t remote_id,
                     const UpdateEdgesRequest* req,
                     UpdateEdgesResponse* res) override {
    return error::Unimplemented("Remote UpdateEdges not implemented");
  }

  Status LookupEdges(const LookupEdgesRequest* req,
                     LookupEdgesResponse* res) override {
    int64_t edge_id = 0;
    int64_t src_id = 0;
    LookupEdgesRequest* request = const_cast<LookupEdgesRequest*>(req);
    res->SetSideInfo(storage_->GetSideInfo(), req->Size());
    while (request->Next(&edge_id, &src_id)) {
      res->AppendWeight(storage_->GetEdgeWeight(edge_id));
      res->AppendLabel(storage_->GetEdgeLabel(edge_id));
      res->AppendTimestamp(storage_->GetEdgeTimestamp(edge_id));
      res->AppendAttribute(storage_->GetEdgeAttribute(edge_id).get());
    }
    return Status::OK();
  }

  Status LookupEdges(int32_t remote_id,
                     const LookupEdgesRequest* req,
                     LookupEdgesResponse* res) override {
    return error::Unimplemented("Remote LookupEdges not implemented");
  }

private:
  io::GraphStorage* storage_;
};

Graph* CreateLocalGraph(const std::string& type,
                        const std::string& view_type,
                        const std::string& use_attrs) {
  return new LocalGraph(type, view_type, use_attrs);
}

}  // namespace graphlearn
