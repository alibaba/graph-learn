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
#include "core/graph/storage/adj_matrix.h"
#include "core/graph/storage/storage_mode.h"
#include "core/graph/storage/topo_statics.h"
#include "core/graph/storage/topo_storage.h"

namespace graphlearn {
namespace io {

namespace {

template <typename A, typename B>
void zip(const std::vector<A> &a,
         const std::vector<B> &b,
         std::vector<std::pair<A, B>>* zipped) {
  for (size_t i = 0; i < a.size(); ++i) {
    zipped->push_back(std::make_pair(a[i], b[i]));
  }
}

template <typename A, typename B>
void unzip(const std::vector<std::pair<A, B>> &zipped,
            std::vector<A>* a,
            std::vector<B>* b) {
  for (size_t i = 0; i < a->size(); i++) {
    (*a)[i] = zipped[i].first;
    (*b)[i] = zipped[i].second;
  }
}

}  // anonymous namespace

class MemoryTopoStorage : public TopoStorage {
public:
  MemoryTopoStorage() : adj_matrix_(nullptr), statics_(nullptr) {
    if (IsDataDistributionEnabled()) {
      statics_ = new TopoStatics(&src_indexing_, &dst_indexing_);
    }
  }

  virtual ~MemoryTopoStorage() {
    delete adj_matrix_;
    delete statics_;
  }

  void Add(IdType edge_id, EdgeValue* edge) override {
    src_indexing_.Add(edge->src_id);
    adj_matrix_->Add(edge_id, edge->src_id, edge->dst_id);
    if (IsDataDistributionEnabled()) {
      dst_indexing_.Add(edge->dst_id);
      statics_->Add(edge->src_id, edge->dst_id);
    }
  }

  void Build(EdgeStorage* edges) override {
    adj_matrix_->Build(edges);
    if (IsDataDistributionEnabled()) {
      statics_->Build();
    }
    if (edges->GetSideInfo()->IsTimestamped()) {
      BuildEdgeIndexing(edges);
    }
  }

  IdType GetEdgeId(IdType edge_index) const override {
    if (edge_indexing_.size() > 0 && edge_index < edge_indexing_.size()) {
      return edge_indexing_[edge_index];
    }
    return edge_index;
  }

  Array<IdType> GetNeighbors(IdType src_id) const override {
    return adj_matrix_->GetNeighbors(src_id);
  }

  Array<IdType> GetOutEdges(IdType src_id) const override {
    return adj_matrix_->GetOutEdges(src_id);
  }

  IndexType GetOutDegree(IdType src_id) const override {
    if (IsDataDistributionEnabled()) {
      return statics_->GetOutDegree(src_id);
    } else {
      return 0;
    }
  }

  IndexType GetInDegree(IdType dst_id) const override {
    if (IsDataDistributionEnabled()) {
      return statics_->GetInDegree(dst_id);
    } else {
      return 0;
    }
  }

  const IdArray GetAllSrcIds() const override {
    if (IsDataDistributionEnabled()) {
      return statics_->GetAllSrcIds();
    } else {
      return IdArray(nullptr, 0);
    }
  }

  const IdArray GetAllDstIds() const override {
    if (IsDataDistributionEnabled()) {
      return statics_->GetAllDstIds();
    } else {
      return IdArray(nullptr, 0);
    }
  }

  const IndexArray GetAllOutDegrees() const override {
    if (IsDataDistributionEnabled()) {
      return statics_->GetAllOutDegrees();
    } else {
      return IndexArray{};
    }
  }

  const IndexArray GetAllInDegrees() const override {
    if (IsDataDistributionEnabled()) {
      return statics_->GetAllInDegrees();
    } else {
      return IndexArray{};
    }
  }

private:
  /// Build edge_indexing with time-order.
  void BuildEdgeIndexing(EdgeStorage* edges) {
    auto& timestamps = edges->GetTimestamps();
    edge_indexing_.reserve(edges->Size());
    std::vector<int64_t> ts;
    ts.reserve(edges->Size());
    for (IdType idx = 0; idx < edges->Size(); ++idx) {
      edge_indexing_.push_back(idx);
      ts.push_back(timestamps.at(idx));
    }
    std::vector<std::pair<IdType, int64_t>> zipped;
    zip(edge_indexing_, ts, &zipped);
    std::sort(std::begin(zipped), std::end(zipped),
        [&](const std::pair<IdType, int64_t>& a,
            const std::pair<IdType, int64_t>& b) {
            return a.second < b.second;
        });
    unzip(zipped, &edge_indexing_, &ts);
  }

private:
  AutoIndex    src_indexing_;
  AutoIndex    dst_indexing_;
  IdList       edge_indexing_;
  AdjMatrix*   adj_matrix_;
  TopoStatics* statics_;

  friend TopoStorage* NewMemoryTopoStorage();
  friend TopoStorage* NewCompressedMemoryTopoStorage();
};

TopoStorage* NewMemoryTopoStorage() {
  MemoryTopoStorage* ret = new MemoryTopoStorage();
  ret->adj_matrix_ = NewMemoryAdjMatrix(&(ret->src_indexing_));
  return ret;
}

TopoStorage* NewCompressedMemoryTopoStorage() {
  MemoryTopoStorage* ret = new MemoryTopoStorage();
  ret->adj_matrix_ = NewCompressedMemoryAdjMatrix(&(ret->src_indexing_));
  return ret;
}

}  // namespace io
}  // namespace graphlearn
