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

#include "core/graph/storage/topo_statics.h"

namespace graphlearn {
namespace io {

TopoStatics::TopoStatics(AutoIndex* src_indexing, AutoIndex* dst_indexing)
  : src_indexing_(src_indexing),
    dst_indexing_(dst_indexing) {
}

void TopoStatics::Build() {
  src_id_list_.shrink_to_fit();
  dst_id_list_.shrink_to_fit();
  out_degree_list_.shrink_to_fit();
  in_degree_list_.shrink_to_fit();
}

void TopoStatics::Add(IdType src_id, IdType dst_id) {
  IndexType src_index = src_indexing_->Get(src_id);
  if (src_index < src_id_list_.size()) {
    // has appeared before
    out_degree_list_[src_index]++;
  } else if (src_index == src_id_list_.size()) {
    // new coming
    src_id_list_.push_back(src_id);
    out_degree_list_.push_back(1);
  } else {
    // just ignore other cases
  }

  IndexType dst_index = dst_indexing_->Get(dst_id);
  if (dst_index < dst_id_list_.size()) {
    // has appeared before
    in_degree_list_[dst_index]++;
  } else if (dst_index == dst_id_list_.size()) {
    // new coming
    dst_id_list_.push_back(dst_id);
    in_degree_list_.push_back(1);
  } else {
    // just ignore other cases
  }
}

IndexType TopoStatics::GetOutDegree(IdType src_id) const {
  IndexType src_index = src_indexing_->Get(src_id);
  if (src_index < out_degree_list_.size()) {
    return out_degree_list_[src_index];
  } else {
    return 0;
  }
}

IndexType TopoStatics::GetInDegree(IdType dst_id) const {
  IndexType dst_index = dst_indexing_->Get(dst_id);
  if (dst_index < in_degree_list_.size()) {
    return in_degree_list_[dst_index];
  } else {
    return 0;
  }
}

}  // namespace io
}  // namespace graphlearn
