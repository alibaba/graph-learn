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

#include "graphlearn/include/sampling_request.h"

#include "graphlearn/include/constants.h"

namespace graphlearn {

namespace {
int32_t kReservedSize = 64;
}  // anonymous namespace

SamplingRequest::SamplingRequest()
    : OpRequest(),
      neighbor_count_(0),
      src_ids_(nullptr) {
}

SamplingRequest::SamplingRequest(const std::string& edge_type,
                                 const std::string& strategy,
                                 int32_t neighbor_count)
    : OpRequest(),
      neighbor_count_(neighbor_count),
      src_ids_(nullptr) {
  params_.reserve(4);

  ADD_TENSOR(params_, kEdgeType, kString, 1);
  params_[kEdgeType].AddString(edge_type);

  ADD_TENSOR(params_, kPartitionKey, kString, 1);
  params_[kPartitionKey].AddString(kSrcIds);

  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(strategy);

  ADD_TENSOR(params_, kNeighborCount, kInt32, 1);
  params_[kNeighborCount].AddInt32(neighbor_count);

  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);
}

OpRequest* SamplingRequest::Clone() const {
  SamplingRequest* req = new SamplingRequest(
    EdgeType(), Strategy(), neighbor_count_);
  return req;
}

void SamplingRequest::SerializeTo(void* request) {
  OpRequest::SerializeTo(request);
}

bool SamplingRequest::ParseFrom(const void* request) {
  if (!OpRequest::ParseFrom(request)) {
    return false;
  }
  neighbor_count_ = params_[kNeighborCount].GetInt32(0);
  src_ids_ = &(tensors_[kSrcIds]);
  return true;
}

void SamplingRequest::Set(const int64_t* src_ids,
                          int32_t batch_size) {
  src_ids_->AddInt64(src_ids, src_ids + batch_size);
}

int32_t SamplingRequest::BatchSize() const {
  return src_ids_->Size();
}

const std::string& SamplingRequest::EdgeType() const {
  return params_.at(kEdgeType).GetString(0);
}

const std::string& SamplingRequest::Strategy() const {
  return params_.at(kOpName).GetString(0);
}

const int64_t* SamplingRequest::GetSrcIds() const {
  if (src_ids_) {
    return src_ids_->GetInt64();
  } else {
    return nullptr;
  }
}

SamplingResponse::SamplingResponse()
    : OpResponse(),
      neighbor_count_(0),
      total_neighbor_count_(0),
      neighbors_(nullptr),
      edges_(nullptr),
      degrees_(nullptr) {
}

void SamplingResponse::SerializeTo(void* response) {
  params_[kNeighborCount].AddInt32(total_neighbor_count_);
  OpResponse::SerializeTo(response);
}

bool SamplingResponse::ParseFrom(const void* response) {
  if (!OpResponse::ParseFrom(response)) {
    return false;
  }
  neighbor_count_ = params_[kNeighborCount].GetInt32(0);
  total_neighbor_count_ = params_[kNeighborCount].GetInt32(1);
  neighbors_ = &(tensors_[kNeighborIds]);
  edges_ = &(tensors_[kEdgeIds]);
  degrees_ = &(tensors_[kDegreeKey]);
  return true;
}

void SamplingResponse::Stitch(ShardsPtr<OpResponse> shards) {
  OpResponse::Stitch(shards);
  shards->ResetNext();
  OpResponse* tmp = nullptr;
  int32_t total_neighbor_count = 0;
  int32_t shard_id = 0;
  while (shards->Next(&shard_id, &tmp)) {
    total_neighbor_count +=
      static_cast<SamplingResponse*>(tmp)->TotalNeighborCount();
  }
  total_neighbor_count_ = total_neighbor_count;
  params_[kNeighborCount].Resize(2);
  params_[kNeighborCount].SetInt32(1, total_neighbor_count_);
}

void SamplingResponse::InitNeighborIds(int32_t count) {
  ADD_TENSOR(tensors_, kNeighborIds, kInt64, count);
  neighbors_ = &(tensors_[kNeighborIds]);
}

void SamplingResponse::InitEdgeIds(int32_t count) {
  ADD_TENSOR(tensors_, kEdgeIds, kInt64, count);
  edges_ = &(tensors_[kEdgeIds]);
}

void SamplingResponse::InitDegrees(int32_t count) {
  ADD_TENSOR(tensors_, kDegreeKey, kInt32, count);
  degrees_ = &(tensors_[kDegreeKey]);
}

void SamplingResponse::SetBatchSize(int32_t batch_size) {
  batch_size_ = batch_size;
}

void SamplingResponse::SetNeighborCount(int32_t neighbor_count) {
  ADD_TENSOR(params_, kNeighborCount, kInt32, 2);
  params_[kNeighborCount].AddInt32(neighbor_count);
  neighbor_count_ = neighbor_count;
}

void SamplingResponse::AppendNeighborId(int64_t id) {
  neighbors_->AddInt64(id);
  ++total_neighbor_count_;
}

void SamplingResponse::AppendEdgeId(int64_t id) {
  edges_->AddInt64(id);
}

void SamplingResponse::AppendDegree(int32_t degree) {
  degrees_->AddInt32(degree);
}

void SamplingResponse::FillWith(int64_t neighbor_id, int64_t edge_id) {
  for (int32_t i = 0; i < neighbor_count_; ++i) {
    neighbors_->AddInt64(neighbor_id);
  }

  // edges is optional
  if (edges_ != nullptr) {
    for (int32_t i = 0; i < neighbor_count_; ++i) {
      edges_->AddInt64(edge_id);
    }
  }

  total_neighbor_count_ += neighbor_count_;
}

int64_t* SamplingResponse::GetNeighborIds() {
  if (neighbors_) {
    return const_cast<int64_t*>(neighbors_->GetInt64());
  } else {
    return nullptr;
  }
}

int64_t* SamplingResponse::GetEdgeIds() {
  if (edges_) {
    return const_cast<int64_t*>(edges_->GetInt64());
  } else {
    return nullptr;
  }
}

int32_t* SamplingResponse::GetDegrees() {
  if (degrees_) {
    return const_cast<int32_t*>(degrees_->GetInt32());
  } else {
    return nullptr;
  }
}

const int64_t* SamplingResponse::GetNeighborIds() const {
  if (neighbors_) {
    return neighbors_->GetInt64();
  } else {
    return nullptr;
  }
}

const int64_t* SamplingResponse::GetEdgeIds() const {
  if (edges_) {
    return edges_->GetInt64();
  } else {
    return nullptr;
  }
}

const int32_t* SamplingResponse::GetDegrees() const {
  if (degrees_) {
    return degrees_->GetInt32();
  } else {
    return nullptr;
  }
}

#define REGISTER_SAMPING_REQUEST(Type) \
  REGISTER_REQUEST(Type##Sampler, SamplingRequest, SamplingResponse)

REGISTER_SAMPING_REQUEST(Random)
REGISTER_SAMPING_REQUEST(Topk)
REGISTER_SAMPING_REQUEST(EdgeWeight)
REGISTER_SAMPING_REQUEST(InDegree)
REGISTER_SAMPING_REQUEST(Full)
REGISTER_SAMPING_REQUEST(RandomNegative)
REGISTER_SAMPING_REQUEST(InDegreeNegative)
REGISTER_SAMPING_REQUEST(SoftInDegreeNegative)

#undef REGISTER_SAMPING_REQUEST

}  // namespace graphlearn
