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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_EDGE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_EDGE_STORAGE_H_

#include <cstdint>
#include <vector>
#include "graphlearn/core/graph/storage/types.h"

namespace graphlearn {
namespace io {

class EdgeStorage {
public:
  virtual ~EdgeStorage() = default;

  virtual void SetSideInfo(const SideInfo* info) = 0;
  virtual const SideInfo* GetSideInfo() const = 0;

  /// Do some re-organization after data fixed.
  virtual void Build() = 0;

  /// Get the total edge count after data fixed.
  virtual IdType Size() const = 0;

  /// An EDGE is made up of [ src_id, dst_id, weight, label, attributes ].
  /// Insert the value to get an unique id.
  /// If the value is invalid, return -1.
  virtual IdType Add(EdgeValue* value) = 0;

  /// Lookup edge infos by edge_id, including
  ///    source node id,
  ///    destination node id,
  ///    edge weight,
  ///    edge label,
  ///    edge attributes
  virtual IdType GetSrcId(IdType edge_id) const = 0;
  virtual IdType GetDstId(IdType edge_id) const = 0;
  virtual float GetWeight(IdType edge_id) const = 0;
  virtual int32_t GetLabel(IdType edge_id) const = 0;
  virtual Attribute GetAttribute(IdType edge_id) const = 0;

  /// For the needs of traversal and sampling, the data distribution is
  /// helpful. The interface should make it convenient to get the global data.
  ///
  /// Get all the source node ids, the count of which is the same with Size().
  /// These ids are not distinct.
  virtual const IdArray GetSrcIds() const = 0;
  /// Get all the destination node ids, the count of which is the same with
  /// Size(). These ids are not distinct.
  virtual const IdArray GetDstIds() const = 0;
  /// Get all weights if existed, the count of which is the same with Size().
  virtual const Array<float> GetWeights() const = 0;
  /// Get all labels if existed, the count of which is the same with Size().
  virtual const Array<int32_t> GetLabels() const = 0;
  /// Get all attributes if existed, the count of which is the same with Size().
  virtual const std::vector<Attribute>* GetAttributes() const = 0;
};

EdgeStorage* NewMemoryEdgeStorage();
EdgeStorage* NewCompressedMemoryEdgeStorage();

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_EDGE_STORAGE_H_
