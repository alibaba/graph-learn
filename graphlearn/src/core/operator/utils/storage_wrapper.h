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

#ifndef GRAPHLEARN_CORE_OPERATOR_UTILS_STORAGE_WRAPPER_H_
#define GRAPHLEARN_CORE_OPERATOR_UTILS_STORAGE_WRAPPER_H_

#include "common/threading/sync/lock.h"
#include "core/graph/graph_store.h"
#include "core/graph/storage/node_storage.h"
#include "core/graph/storage/types.h"

namespace graphlearn {
namespace op {

class StorageWrapper {
public:
  StorageWrapper(NodeFrom node_from, const std::string& type,
      GraphStore* graph_store);
  ~StorageWrapper() {}

  const ::graphlearn::io::IdArray GetIds();
  const ::graphlearn::io::Array<float> GetNodeWeights() const;
  const ::graphlearn::io::IndexArray GetAllInDegrees() const;
  ::graphlearn::io::Array<int64_t> GetNeighbors(int64_t src_id) const;
  void Lock(); 
  void Unlock();
  const std::string& Type() const;
  NodeFrom From() const;

private:
  ::graphlearn::io::NodeStorage*  node_storage_;
  ::graphlearn::io::GraphStorage* graph_storage_;
  NodeFrom                        node_from_;
};

}  // namespace op
}  // namespace graphlearn

#endif //  GRAPHLEARN_CORE_OPERATOR_UTILS_STORAGE_WRAPPER_H_
