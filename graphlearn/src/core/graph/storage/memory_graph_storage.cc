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

#include <algorithm>
#include <functional>
#include <mutex>  // NOLINT [build/c++11]

#include "common/threading/sync/lock.h"
#include "core/graph/storage/edge_storage.h"
#include "core/graph/storage/graph_storage.h"
#include "core/graph/storage/topo_storage.h"

namespace graphlearn {
namespace io {

class MemoryGraphStorage : public GraphStorage {
public:
  MemoryGraphStorage() {
    topo_ = NewMemoryTopoStorage();
    edges_ = NewMemoryEdgeStorage();
  }

  virtual ~MemoryGraphStorage() {
    delete topo_;
    delete edges_;
  }

  void Lock() override {
    mtx_.lock();
  }

  void Unlock() override {
    mtx_.unlock();
  }

  void Add(EdgeValue* edge) override {
    IdType edge_id = edges_->Add(edge);
    if (edge_id != -1) {
      topo_->Add(edge_id, edge);
    }
  }

  void Build() override {
    ScopedLocker<std::mutex> _(&mtx_);
    edges_->Build();
    topo_->Build(edges_);
  }

  void SetSideInfo(const SideInfo* info) override {
    return edges_->SetSideInfo(info);
  }

  const SideInfo* GetSideInfo() const override {
    return edges_->GetSideInfo();
  }

  IdType GetEdgeCount() const override {
    return edges_->Size();
  }

  IdType GetSrcId(IdType edge_id) const override {
    return edges_->GetSrcId(edge_id);
  }

  IdType GetDstId(IdType edge_id) const override {
    return edges_->GetDstId(edge_id);
  }

  IdType GetEdgeId(IdType edge_index) const override {
    return topo_->GetEdgeId(edge_index);
  }

  int32_t GetEdgeLabel(IdType edge_id) const override {
    return edges_->GetLabel(edge_id);
  }

  int64_t GetEdgeTimestamp(IdType edge_id) const override {
    return edges_->GetTimestamp(edge_id);
  }

  float GetEdgeWeight(IdType edge_id) const override {
    return edges_->GetWeight(edge_id);
  }

  Attribute GetEdgeAttribute(IdType edge_id) const override {
    return edges_->GetAttribute(edge_id);
  }

  Array<IdType> GetNeighbors(IdType src_id) const override {
    return topo_->GetNeighbors(src_id);
  }

  Array<IdType> GetOutEdges(IdType src_id) const override {
    return topo_->GetOutEdges(src_id);
  }

  IndexType GetInDegree(IdType dst_id) const override {
    return topo_->GetInDegree(dst_id);
  }

  IndexType GetOutDegree(IdType src_id) const override {
    return topo_->GetOutDegree(src_id);
  }

  const IndexArray GetAllInDegrees() const override {
    return topo_->GetAllInDegrees();
  }

  const IndexArray GetAllOutDegrees() const override {
    return topo_->GetAllOutDegrees();
  }

  const IdArray GetAllSrcIds() const override {
    return topo_->GetAllSrcIds();
  }

  const IdArray GetAllDstIds() const override {
    return topo_->GetAllDstIds();
  }

private:
  std::mutex   mtx_;
  EdgeStorage* edges_;
  TopoStorage* topo_;
};

GraphStorage* NewMemoryGraphStorage() {
  return new MemoryGraphStorage();
}

}  // namespace io
}  // namespace graphlearn
