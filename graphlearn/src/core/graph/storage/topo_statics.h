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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_TOPO_STATICS_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_TOPO_STATICS_H_

#include <cstdint>
#include <vector>
#include "core/graph/storage/auto_indexing.h"
#include "core/graph/storage/types.h"

namespace graphlearn {
namespace io {

class TopoStatics {
public:
  TopoStatics(AutoIndex* src_indexing, AutoIndex* dst_indexing);
  ~TopoStatics() = default;

  void Build();

  void Add(IdType src_id, IdType dst_id);

  const IdArray GetAllSrcIds() const {
    return IdArray(src_id_list_.data(), src_id_list_.size());
  }

  const IdArray GetAllDstIds() const {
    return IdArray(dst_id_list_.data(), dst_id_list_.size());
  }

  const IndexList* GetAllOutDegrees() const {
    return &out_degree_list_;
  }

  const IndexList* GetAllInDegrees() const {
    return &in_degree_list_;
  }

  IndexType GetOutDegree(IdType src_id) const;
  IndexType GetInDegree(IdType dst_id) const;

private:
  AutoIndex* src_indexing_;
  AutoIndex* dst_indexing_;
  IdList     src_id_list_;
  IdList     dst_id_list_;
  IndexList  out_degree_list_;
  IndexList  in_degree_list_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_TOPO_STATICS_H_
