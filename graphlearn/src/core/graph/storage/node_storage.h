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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_NODE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_NODE_STORAGE_H_

#include <cstdint>
#include <vector>
#include "core/graph/storage/types.h"

namespace graphlearn {
namespace io {

class NodeStorage {
public:
  virtual ~NodeStorage() = default;

  virtual void Lock() = 0;
  virtual void Unlock() = 0;

  virtual void SetSideInfo(const SideInfo* info) = 0;
  virtual const SideInfo* GetSideInfo() const = 0;

  /// Do some re-organization after data fixed.
  virtual void Build() = 0;

  /// Get the total node count after data fixed.
  virtual IdType Size() const = 0;

  /// A NODE is made up of [ id, attributes, weight, label ].
  /// Insert a node. If a node with the same id existed, just ignore.
  virtual void Add(NodeValue* value) = 0;

  /// Lookup node infos by node_id, including
  ///    node weight,
  ///    node label,
  ///    node attributes
  virtual float GetWeight(IdType node_id) const = 0;
  virtual int32_t GetLabel(IdType node_id) const = 0;
  virtual Attribute GetAttribute(IdType node_id) const = 0;

  /// For the needs of traversal and sampling, the data distribution is
  /// helpful. The interface should make it convenient to get the global data.
  ///
  /// Get all the node ids, the count of which is the same with Size().
  /// These ids are distinct.
  virtual const IdArray GetIds() const = 0;
  /// Get all weights if existed, the count of which is the same with Size().
  virtual const Array<float> GetWeights() const = 0;
  /// Get all labels if existed, the count of which is the same with Size().
  virtual const Array<int32_t> GetLabels() const = 0;
  /// Get all attributes if existed, the count of which is the same with Size().
  virtual const std::vector<Attribute>* GetAttributes() const = 0;
};

NodeStorage* NewMemoryNodeStorage();
NodeStorage* NewCompressedMemoryNodeStorage();
NodeStorage* NewVineyardNodeStorage(
    const std::string& node_type,
    const std::string& view_type,
    const std::string& use_attrs);

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_NODE_STORAGE_H_
