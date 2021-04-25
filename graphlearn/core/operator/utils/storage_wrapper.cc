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

#include "graphlearn/core/operator/utils/storage_wrapper.h"

namespace graphlearn {
namespace op {

StorageWrapper::StorageWrapper(NodeFrom node_from, const std::string& type,
    GraphStore* graph_store) : node_from_(node_from) {
  if (node_from == NodeFrom::kNode) {
    Noder* noder = graph_store->GetNoder(type);
    node_storage_ = noder->GetLocalStorage();
    graph_storage_ = nullptr;
  } else {
    Graph* graph = graph_store->GetGraph(type);
    graph_storage_ = graph->GetLocalStorage();
    node_storage_ = nullptr;
  }
}

const ::graphlearn::io::IdList* StorageWrapper::GetIds() {
  if (node_from_ == NodeFrom::kNode) {
    return node_storage_->GetIds();
  } else {
    if (node_from_ == NodeFrom::kEdgeSrc) {
      return graph_storage_->GetAllSrcIds();
    } else {
      return graph_storage_->GetAllDstIds();
    }
  }
}

const std::vector<float>* StorageWrapper::GetNodeWeights() const {
  if (node_storage_ != nullptr) {
    return node_storage_->GetWeights();
  } else {
    return nullptr;
  }
}

const std::vector<int32_t>* StorageWrapper::GetAllInDegrees() const {
  if (graph_storage_ != nullptr) {
    return graph_storage_->GetAllInDegrees();
  } else {
    return nullptr;
  }
}

::graphlearn::io::Array<int64_t> StorageWrapper::GetNeighbors(int64_t src_id) const {
  if (graph_storage_ != nullptr) {
    return graph_storage_->GetNeighbors(src_id);
  } else {
    return ::graphlearn::io::Array<int64_t>();
  }
}

void StorageWrapper::Lock() {
  if (node_storage_ != nullptr) {
    node_storage_->Lock();
  } else {
    graph_storage_->Lock();
  }
}

void StorageWrapper::Unlock() {
  if (node_storage_ != nullptr) {
    node_storage_->Unlock();
  } else {
    graph_storage_->Unlock();
  }
}

const std::string& StorageWrapper::Type() const {
  if (node_storage_ != nullptr) {
    return node_storage_->GetSideInfo()->type;
  } else {
    return graph_storage_->GetSideInfo()->type;
  }
}

NodeFrom StorageWrapper::From() const {
  return node_from_;
}


}  // namespace op
}  // namespace graphlearn
