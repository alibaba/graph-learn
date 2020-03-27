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
#include "graphlearn/core/graph/storage/types.h"

namespace graphlearn {
namespace io {

class NodeStorage {
public:
  virtual ~NodeStorage() = default;

  virtual void Lock() = 0;
  virtual void Unlock() = 0;

  virtual void Build() = 0;

  virtual void SetSideInfo(const SideInfo* info) = 0;
  virtual const SideInfo* GetSideInfo() const = 0;

  virtual void Add(NodeValue* value) = 0;
  virtual IdType Size() const = 0;

  virtual IndexType GetLabel(IdType node_id) const = 0;
  virtual float GetWeight(IdType node_id) const = 0;
  virtual const Attribute* GetAttribute(IdType node_id) const = 0;

  virtual const IdList* GetIds() const = 0;
  virtual const IndexList* GetLabels() const = 0;
  virtual const std::vector<float>* GetWeights() const = 0;
  virtual const std::vector<Attribute*>* GetAttributes() const = 0;
};

NodeStorage* NewMemoryNodeStorage();

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_NODE_STORAGE_H_
