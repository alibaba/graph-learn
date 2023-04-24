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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_TOPO_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_TOPO_STORAGE_H_

#include <cstdint>
#include <string>
#include <vector>
#include "core/graph/storage/edge_storage.h"
#include "core/graph/storage/types.h"

namespace graphlearn {
namespace io {

class TopoStorage {
public:
  virtual ~TopoStorage() = default;

  /// Do some re-organization after data fixed.
  virtual void Build(EdgeStorage* edges) = 0;

  /// An EDGE is made up of [ src_id, attributes, dst_id ].
  /// Before inserted to the TopoStorage, it should be inserted to
  /// EdgeStorage to get an unique id. And then use the id and value here.
  virtual void Add(IdType edge_id, EdgeValue* value) = 0;

  /// The original edge_id and edge_index are consistent.
  /// In temporal graph, the edge_indexes are re-organized according to the
  /// time-order in `Build(EdgeStorage* edges)`.
  virtual IdType GetEdgeId(IdType edge_index) const = 0;

  /// Get all the neighbor node ids of a given id.
  virtual Array<IdType> GetNeighbors(IdType src_id) const = 0;
  /// Get all the neighbor edge ids of a given id.
  virtual Array<IdType> GetOutEdges(IdType src_id) const = 0;
  /// Get the out-degree value of a given id.
  virtual IndexType GetOutDegree(IdType src_id) const = 0;
  /// Get the in-degree value of a given id.
  virtual IndexType GetInDegree(IdType dst_id) const = 0;

  /// Get all the distinct ids that appear as the source id of an edge.
  /// For example, 6 edges like
  /// [1 2]
  /// [2 3]
  /// [2 4]
  /// [1 3]
  /// [3 1]
  /// [3 2]
  /// GetAllSrcIds() --> {1, 2, 3}
  virtual const IdArray GetAllSrcIds() const = 0;

  /// Get all the distinct ids that appear as the destination id of an edge.
  /// For the above example, GetAllDstIds() --> {2, 3, 4, 1}
  virtual const IdArray GetAllDstIds() const = 0;

  /// Get the out-degree values of all ids corresponding to GetAllSrcIds().
  /// For the above example, GetAllOutDegrees() --> {2, 2, 2}
  virtual const IndexArray GetAllOutDegrees() const = 0;

  /// Get the in-degree values of all ids corresponding to GetAllDstIds().
  /// For the above example, GetAllInDegrees() --> {2, 2, 1, 1}
  virtual const IndexArray GetAllInDegrees() const = 0;
};

TopoStorage* NewMemoryTopoStorage();
TopoStorage* NewCompressedMemoryTopoStorage();

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_TOPO_STORAGE_H_
