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
#include "core/graph/storage/adj_matrix.h"

namespace graphlearn {
namespace io {

namespace {

template <typename A, typename B, typename C>
void zip3(const std::vector<A> &a,
          const std::vector<B> &b,
          const std::vector<C> &c,
          std::vector<std::pair<std::pair<A, B>, C>>* zipped) {
  for (size_t i = 0; i < a.size(); ++i) {
    zipped->push_back(std::make_pair(std::make_pair(a[i], b[i]), c[i]));
  }
}

template <typename A, typename B, typename C>
void unzip3(const std::vector<std::pair<std::pair<A, B>, C>> &zipped,
            std::vector<A>* a,
            std::vector<B>* b,
            std::vector<C>* c) {
  for (size_t i = 0; i < a->size(); i++) {
    (*a)[i] = zipped[i].first.first;
    (*b)[i] = zipped[i].first.second;
    (*c)[i] = zipped[i].second;
  }
}

}  // anonymous namespace

class CompressedMemoryAdjMatrix;

class MemoryAdjMatrix : public AdjMatrix {
public:
  explicit MemoryAdjMatrix(AutoIndex* indexing)
    : indexing_(indexing) {
  }

  virtual ~MemoryAdjMatrix() {
  }

  void Build(EdgeStorage* edges) override {
    if (edges->GetSideInfo()->IsWeighted()) {
      Sort(edges);
    }
  }

  IdType Size() const override {
    return adj_nodes_.size();
  }

  void Add(IdType edge_id, IdType src_id, IdType dst_id) override {
    IndexType src_index = indexing_->Get(src_id);

    if (src_index < adj_nodes_.size()) {
      adj_nodes_[src_index].emplace_back(dst_id);
      adj_edges_[src_index].emplace_back(edge_id);
    } else {
      std::vector<IdType> neighbors(1, dst_id);
      adj_nodes_.push_back(std::move(neighbors));

      std::vector<IdType> edge_ids(1, edge_id);
      adj_edges_.push_back(std::move(edge_ids));
    }
  }

  Array<IdType> GetNeighbors(IdType src_id) const override {
    return LookupFrom(src_id, adj_nodes_);
  }

  Array<IdType> GetOutEdges(IdType src_id) const override {
    return LookupFrom(src_id, adj_edges_);
  }

private:
  Array<IdType> LookupFrom(IdType src_id, const IdMatrix& from) const {
    IndexType index = indexing_->Get(src_id);
    if (index == -1) {
      return Array<IdType>();
    } else {
      return Array<IdType>(from[index]);
    }
  }

  void Sort(EdgeStorage* edges) {
    for (IndexType i = 0; i < adj_nodes_.size(); ++i) {
      auto& dst_ids = adj_nodes_[i];
      auto& edge_ids = adj_edges_[i];

      std::vector<float> weights;
      weights.reserve(edge_ids.size());
      for (auto edge_id : edge_ids) {
        weights.push_back(edges->GetWeight(edge_id));
      }

      std::vector<std::pair<std::pair<IdType, IdType>, float>> zipped;
      zip3(dst_ids, edge_ids, weights, &zipped);
      std::sort(std::begin(zipped), std::end(zipped),
          [&](const std::pair<std::pair<IdType, IdType>, float>& a,
              const std::pair<std::pair<IdType, IdType>, float>& b) {
              return a.second > b.second;
          });
      unzip3(zipped, &dst_ids, &edge_ids, &weights);
    }
  }

private:
  AutoIndex* indexing_;
  IdMatrix adj_nodes_;
  IdMatrix adj_edges_;

  friend class CompressedMemoryAdjMatrix;
};

class CompressedMemoryAdjMatrix : public AdjMatrix {
public:
  explicit CompressedMemoryAdjMatrix(AutoIndex* indexing)
      : naive_adj_(nullptr), indexing_(indexing) {
    naive_adj_.reset(new MemoryAdjMatrix(indexing));
  }

  virtual ~CompressedMemoryAdjMatrix() {
  }

  void Build(EdgeStorage* edges) override {
    naive_adj_->Build(edges);

    offsets_.push_back(0);
    auto& node_ids = naive_adj_->adj_nodes_;
    auto& edge_ids = naive_adj_->adj_edges_;
    for (size_t i = 0; i < node_ids.size(); ++i) {
      for (size_t j = 0; j < node_ids[i].size(); ++j) {
        adj_nodes_.push_back(node_ids[i][j]);
        adj_edges_.push_back(edge_ids[i][j]);
      }
      offsets_.push_back(adj_nodes_.size());

      node_ids[i].clear();
      edge_ids[i].clear();
    }

    node_ids.clear();
    edge_ids.clear();
    naive_adj_.reset();
  }

  IdType Size() const override {
    return offsets_.size() - 1;
  }

  void Add(IdType edge_id, IdType src_id, IdType dst_id) override {
    naive_adj_->Add(edge_id, src_id, dst_id);
  }

  Array<IdType> GetNeighbors(IdType src_id) const override {
    return LookupFrom(src_id, adj_nodes_);
  }

  Array<IdType> GetOutEdges(IdType src_id) const override {
    return LookupFrom(src_id, adj_edges_);
  }

private:
  Array<IdType> LookupFrom(IdType src_id, const IdList& from) const {
    IndexType index = indexing_->Get(src_id);
    if (index == -1) {
      return Array<IdType>();
    } else {
      IndexType offset = offsets_[index];
      IndexType size = offsets_[index + 1] - offset;
      return Array<IdType>(from.data() + offset, size);
    }
  }

private:
  std::unique_ptr<MemoryAdjMatrix> naive_adj_;
  AutoIndex* indexing_;
  IndexList offsets_;
  IdList adj_nodes_;
  IdList adj_edges_;
};

AdjMatrix* NewMemoryAdjMatrix(AutoIndex* indexing) {
  return new MemoryAdjMatrix(indexing);
}

AdjMatrix* NewCompressedMemoryAdjMatrix(AutoIndex* indexing) {
  return new CompressedMemoryAdjMatrix(indexing);
}

}  // namespace io
}  // namespace graphlearn
