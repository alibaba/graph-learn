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

#include "graphlearn/core/operator/aggregator/aggregator.h"

#include "graphlearn/core/graph/storage/node_storage.h"
#include "graphlearn/include/config.h"
#include "graphlearn/common/base/log.h"

namespace graphlearn {
namespace op {

Status Aggregator::Aggregate(const AggregatingRequest* req,
                             AggregatingResponse* res) {
  Noder* node = graph_store_->GetNoder(req->Type());
  ::graphlearn::io::NodeStorage* storage = node->GetLocalStorage();
  const ::graphlearn::io::SideInfo* info = storage->GetSideInfo();

  int32_t dim = info->f_num;
  res->SetEmbeddingDim(dim);
  int32_t num_segments = req->NumSegments();
  res->SetNumSegments(num_segments);
  res->SetName(req->Name());

  // Initialize float attributes.
  // Aggregator only takes effect on float attributes.
  float* emb = new float[dim]();

  int64_t node_id = 0;
  int32_t segment_id = 0;
  int32_t segment_size = 0;
  AggregatingRequest* request = const_cast<AggregatingRequest*>(req);
  for (int32_t idx = 0; idx < num_segments; idx++) {
    segment_size = 0;
    this->InitFunc(emb, dim);
    while (!req->SegmentEnd(idx)) {
      request->Next(&node_id, &segment_id);
      segment_size++;
      this->AggFunc(
        emb, storage->GetAttribute(node_id)->f_attrs.data(), dim);
    }
    this->FinalFunc(emb, dim, &segment_size, 1);
    res->AppendEmbedding(emb);
    res->AppendSegment(segment_size);
  }
  return Status::OK();
}

void Aggregator::InitFunc(float* value, int32_t size) {
  for (int32_t i = 0; i < size; ++i) {
    value[i] = 0.0;
  }
}

void Aggregator::AggFunc(float* left,
                         const float* right,
                         int32_t size,
                         const int32_t* segments,
                         int32_t num_segments) {
}

void Aggregator::FinalFunc(float* values,
                           int32_t size,
                           const int32_t* segments,
                           int32_t num_segments) {
  int32_t dim = size / num_segments;
  for (int32_t idx = 0; idx < num_segments; ++idx) {
    if (segments[idx] == 0) {
      for (int32_t i = 0; i < dim; ++i) {
        values[idx * dim + i] = GLOBAL_FLAG(DefaultFloatAttribute);
      }
    }
  }
}

}  // namespace op
}  // namespace graphlearn
