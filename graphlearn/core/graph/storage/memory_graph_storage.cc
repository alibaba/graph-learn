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

#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/core/graph/storage/edge_storage.h"
#include "graphlearn/core/graph/storage/graph_storage.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace io {

class MemoryGraphStorage : public GraphStorage {
public:
  MemoryGraphStorage() {
    int64_t estimate_size = GLOBAL_FLAG(AverageNodeCount);

    src_id_index_.rehash(estimate_size);
    dst_id_index_.rehash(estimate_size);
    src_id_list_.reserve(estimate_size);
    dst_id_list_.reserve(estimate_size);
    adj_matrix_.reserve(estimate_size);
    adj_edge_matrix_.reserve(estimate_size);
    src_degree_list_.reserve(estimate_size);
    dst_degree_list_.reserve(estimate_size);

    edges_ = NewMemoryEdgeStorage();
  }

  virtual ~MemoryGraphStorage() {
    delete edges_;
  }

  void Lock() override {
    mtx_.lock();
  }

  void Unlock() override {
    mtx_.unlock();
  }

  void Add(EdgeValue* edge) override {
    // Insert edge
    IdType edge_id = edges_->Add(edge);

    // Insert src node
    IndexType src_index = src_id_index_.size();
    auto ret = src_id_index_.insert({edge->src_id, src_index});
    if (ret.second) {
      src_id_list_.push_back(edge->src_id);
      src_degree_list_.push_back(1);
    } else {
      src_index = ret.first->second;
      src_degree_list_[src_index]++;
    }

    // Insert dst node
    IndexType dst_index = dst_id_index_.size();
    ret = dst_id_index_.insert({edge->dst_id, dst_index});
    if (ret.second) {
      dst_id_list_.push_back(edge->dst_id);
      dst_degree_list_.push_back(1);
    } else {
      dst_index = ret.first->second;
      dst_degree_list_[dst_index]++;
    }

    // Insert adjacent matrix
    if (src_index < adj_matrix_.size()) {
      adj_matrix_[src_index].emplace_back(edge->dst_id);
      adj_edge_matrix_[src_index].emplace_back(edge_id);
    } else {
      std::vector<IdType> neighbors(1, edge->dst_id);
      adj_matrix_.push_back(std::move(neighbors));

      std::vector<IdType> edge_ids(1, edge_id);
      adj_edge_matrix_.push_back(std::move(edge_ids));
    }
  }

  void Build() override {
    Resize(); // resize vectors;
    Sort(); // sort edges by edge weight;
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

  IndexType GetEdgeLabel(IdType edge_id) const override {
    return edges_->GetLabel(edge_id);
  }

  float GetEdgeWeight(IdType edge_id) const override {
    return edges_->GetWeight(edge_id);
  }

  const Attribute* GetEdgeAttribute(IdType edge_id) const override {
    return edges_->GetAttribute(edge_id);
  }

  const IdList* GetNeighbors(IdType src_id) const override {
    auto it = src_id_index_.find(src_id);
    if (it == src_id_index_.end()) {
      return nullptr;
    } else {
      return &(adj_matrix_[it->second]);
    }
  }

  const IdList* GetOutEdges(IdType src_id) const override {
    auto it = src_id_index_.find(src_id);
    if (it == src_id_index_.end()) {
      return nullptr;
    } else {
      return &(adj_edge_matrix_[it->second]);
    }
  }

  IndexType GetInDegree(IdType dst_id) const override {
    auto it = dst_id_index_.find(dst_id);
    if (it == dst_id_index_.end()) {
      return 0;
    } else {
      return dst_degree_list_[it->second];
    }
  }

  IndexType GetOutDegree(IdType src_id) const override {
    auto it = src_id_index_.find(src_id);
    if (it == src_id_index_.end()) {
      return 0;
    } else {
      return src_degree_list_[it->second];
    }
  }

  const IndexList* GetAllInDegrees() const override {
    return &dst_degree_list_;
  }

  const IndexList* GetAllOutDegrees() const override {
    return &src_degree_list_;
  }

  const IdList* GetAllSrcIds() const override {
    return &src_id_list_;
  }

  const IdList* GetAllDstIds() const override {
    return &dst_id_list_;
  }

private:
  void Resize() {
    ScopedLocker<std::mutex> _(&mtx_);
    src_id_list_.shrink_to_fit();
    dst_id_list_.shrink_to_fit();
    src_degree_list_.shrink_to_fit();
    dst_degree_list_.shrink_to_fit();
    adj_matrix_.shrink_to_fit();
    adj_edge_matrix_.shrink_to_fit();
  }

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

  void Sort() {
    ScopedLocker<std::mutex> _(&mtx_);
    if (GetSideInfo()->IsWeighted()) {
      for (auto& it: src_id_index_) {
        auto& dst_ids = adj_matrix_[it.second];
        auto& edge_ids = adj_edge_matrix_[it.second];

        std::vector<float> weights;
        for (auto edge_id: edge_ids) {
          weights.push_back(GetEdgeWeight(edge_id));
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
  }


private:
  std::mutex   mtx_;
  EdgeStorage* edges_;

  MAP       src_id_index_;
  MAP       dst_id_index_;
  IdList    src_id_list_;
  IdList    dst_id_list_;
  IndexList src_degree_list_;
  IndexList dst_degree_list_;
  IdMatrix  adj_matrix_;
  IdMatrix  adj_edge_matrix_;
};

GraphStorage* NewMemoryGraphStorage() {
  return new MemoryGraphStorage();
}

}  // namespace io
}  // namespace graphlearn
