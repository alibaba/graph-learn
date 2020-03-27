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

#include "graphlearn/include/aggregation_request.h"

#include "graphlearn/include/constants.h"

namespace graphlearn {

AggregateNodesRequest::AggregateNodesRequest()
    : OpRequest(), id_cursor_(0), segment_cursor_(0) {
}

AggregateNodesRequest::AggregateNodesRequest(const std::string& node_type,
                                             const std::string& strategy)
    : OpRequest(), id_cursor_(0), segment_cursor_(0) {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(strategy);

  ADD_TENSOR(params_, kPartitionKey, kString, 1);
  params_[kPartitionKey].AddString(kNodeIds);

  ADD_TENSOR(params_, kNodeType, kString, 1);
  params_[kNodeType].AddString(node_type);
}

OpRequest* AggregateNodesRequest::Clone() const {
  AggregateNodesRequest* req = new AggregateNodesRequest();
  return req;
}

void AggregateNodesRequest::SerializeTo(void* request) {
  ADD_TENSOR(params_, kBatchSize, kInt32, 2);
  params_[kBatchSize].AddInt32(num_ids_);
  params_[kBatchSize].AddInt32(num_segments_);
  OpRequest::SerializeTo(request);
}

bool AggregateNodesRequest::ParseFrom(const void* request) {
  if (!OpRequest::ParseFrom(request)) {
    return false;
  }
  num_ids_ = params_[kBatchSize].GetInt32(0);
  num_segments_ = params_[kBatchSize].GetInt32(1);
  node_ids_ = &(tensors_[kNodeIds]);
  segments_ = &(tensors_[kDegreeKey]);
  return true;
}

void AggregateNodesRequest::Set(const int64_t* node_ids,
                                int32_t num_ids,
                                const int32_t* segments,
                                int32_t num_segments) {
  ADD_TENSOR(tensors_, kNodeIds, kInt64, num_ids);
  node_ids_ = &(tensors_[kNodeIds]);
  node_ids_->AddInt64(node_ids, node_ids + num_ids);
  num_ids_ = num_ids;

  ADD_TENSOR(tensors_, kDegreeKey, kInt32, num_segments);
  segments_ = &(tensors_[kDegreeKey]);
  segments_->AddInt32(segments, segments + num_segments);
  num_segments_ = num_segments;
}

const std::string& AggregateNodesRequest::NodeType() const {
  return params_.at(kNodeType).GetString(0);
}

bool AggregateNodesRequest::NextId(int64_t* node_id) {
  if (id_cursor_ >= num_ids_) {
    return false;
  }

  *node_id = node_ids_->GetInt64(id_cursor_);
  ++id_cursor_;
  return true;
}

bool AggregateNodesRequest::NextSegment(int32_t* segment) {
  if (segment_cursor_ >= num_segments_) {
    return false;
  }

  *segment = segments_->GetInt32(segment_cursor_);
  ++segment_cursor_;
  return true;
}

AggregateNodesResponse::AggregateNodesResponse()
    : LookupResponse() {
}

REGISTER_REQUEST(MinAggregator, AggregateNodesRequest, AggregateNodesResponse);
REGISTER_REQUEST(ProdAggregator, AggregateNodesRequest, AggregateNodesResponse);
REGISTER_REQUEST(SumAggregator, AggregateNodesRequest, AggregateNodesResponse);
REGISTER_REQUEST(MaxAggregator, AggregateNodesRequest, AggregateNodesResponse);
REGISTER_REQUEST(MeanAggregator, AggregateNodesRequest, AggregateNodesResponse);

}  // namespace graphlearn
