/* Copyright 2020-2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_TOPO_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_TOPO_STORAGE_H_

#if defined(WITH_VINEYARD)
#include "vineyard/graph/fragment/arrow_fragment.vineyard.h"
#include "vineyard/graph/fragment/arrow_fragment_group.h"
#endif

#include "core/graph/storage/topo_storage.h"
#include "core/graph/storage/vineyard_graph_storage.h"
#include "include/config.h"

#if defined(WITH_VINEYARD)

namespace graphlearn {
namespace io {

class VineyardTopoStorage : public graphlearn::io::TopoStorage {
public:
  explicit VineyardTopoStorage(std::string edge_label = "0",
                               const std::string& decorated_edge_view = "",
                               const std::string& use_attrs = "") {
    graph_ = new VineyardGraphStorage(edge_label, decorated_edge_view, use_attrs);
  }

  virtual ~VineyardTopoStorage() {
    delete graph_;
  };

  /// Do some re-organization after data fixed.
  virtual void Build(EdgeStorage *edges) override {}

  /// An EDGE is made up of [ src_id, attributes, dst_id ].
  /// Before inserted to the TopoStorage, it should be inserted to
  /// EdgeStorage to get an unique id. And then use the id and value here.
  virtual void Add(IdType edge_id, EdgeValue *value) override {}

  /// Get all the neighbor node ids of a given id.
  virtual Array<IdType> GetNeighbors(IdType src_id) const override {
    return graph_->GetNeighbors(src_id);
  }
  /// Get all the neighbor edge ids of a given id.
  virtual Array<IdType> GetOutEdges(IdType src_id) const override {
    return graph_->GetOutEdges(src_id);
  }
  /// Get the in-degree value of a given id.
  virtual IndexType GetInDegree(IdType dst_id) const override {
    return graph_->GetInDegree(dst_id);
  }
  /// Get the out-degree value of a given id.
  virtual IndexType GetOutDegree(IdType src_id) const override {
    return graph_->GetOutDegree(src_id);
  }

  /// Get all the distinct ids that appear as the source id of an edge.
  /// For example, 6 edges like
  /// [1 2]
  /// [2 3]
  /// [2 4]
  /// [1 3]
  /// [3 1]
  /// [3 2]
  /// GetAllSrcIds() --> {1, 2, 3}
  virtual const IdArray GetAllSrcIds() const override {
    return graph_->GetAllSrcIds();
  }

  /// Get all the distinct ids that appear as the destination id of an edge.
  /// For the above example, GetAllDstIds() --> {2, 3, 4, 1}
  virtual const IdArray GetAllDstIds() const override {
    return graph_->GetAllDstIds();
  }

  /// Get the out-degree values of all ids corresponding to GetAllSrcIds().
  /// For the above example, GetAllOutDegrees() --> {2, 2, 2}
  virtual const IndexArray GetAllOutDegrees() const override {
    return graph_->GetAllOutDegrees();
  }

  /// Get the in-degree values of all ids corresponding to GetAllDstIds().
  /// For the above example, GetAllInDegrees() --> {2, 2, 1, 1}
  virtual const IndexArray GetAllInDegrees() const override {
    return graph_->GetAllInDegrees();
  }

private:
  VineyardGraphStorage *graph_ = nullptr;
};

} // namespace io
} // namespace graphlearn

#endif

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_TOPO_STORAGE_H_
