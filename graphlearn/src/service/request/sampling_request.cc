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
      filter_type_(0),
      src_ids_(nullptr),
      filter_ids_(nullptr) {
}

SamplingRequest::SamplingRequest(const std::string& type,
                                 const std::string& strategy,
                                 int32_t neighbor_count,
                                 int32_t filter_type)
    : OpRequest(),
      neighbor_count_(neighbor_count),
      filter_type_(filter_type),
      src_ids_(nullptr),
      filter_ids_(nullptr) {
  params_.reserve(kReservedSize);

  ADD_TENSOR(params_, kType, kString, 1);
  params_[kType].AddString(type);

  ADD_TENSOR(params_, kPartitionKey, kString, 1);
  params_[kPartitionKey].AddString(kSrcIds);

  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(strategy);

  ADD_TENSOR(params_, kNeighborCount, kInt32, 1);
  params_[kNeighborCount].AddInt32(neighbor_count);

  ADD_TENSOR(params_, kFilterType, kInt32, 1);
  params_[kFilterType].AddInt32(filter_type);

  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);

  if (filter_type > 0) {
    // If other filter types to be supported, extend here.
    ADD_TENSOR(tensors_, kFilterIds, kInt64, kReservedSize);
    filter_ids_ = &(tensors_[kFilterIds]);
  }
}

OpRequest* SamplingRequest::Clone() const {
  SamplingRequest* req = new SamplingRequest(
    Type(), Strategy(), neighbor_count_);
  return req;
}

void SamplingRequest::SetMembers() {
  neighbor_count_ = params_[kNeighborCount].GetInt32(0);
  filter_type_ = params_[kFilterType].GetInt32(0);
  src_ids_ = &(tensors_[kSrcIds]);
  if (filter_type_ > 0) {
    filter_ids_ = &(tensors_[kFilterIds]);
  }
}

void SamplingRequest::Init(const Tensor::Map& params) {
  params_.reserve(kReservedSize);
  ADD_TENSOR(params_, kType, kString, 1);
  params_[kType].AddString(params.at(kEdgeType).GetString(0));
  ADD_TENSOR(params_, kPartitionKey, kString, 1);
  params_[kPartitionKey].AddString(kSrcIds);
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(params.at(kStrategy).GetString(0));
  ADD_TENSOR(params_, kNeighborCount, kInt32, 1);
  params_[kNeighborCount].AddInt32(params.at(kNeighborCount).GetInt32(0));

  ADD_TENSOR(params_, kFilterType, kInt32, 1);
  if (params.find(kFilterType) != params.end()) {
    params_[kFilterType].AddInt32(params.at(kFilterType).GetInt32(0));
  } else {
    params_[kFilterType].AddInt32(0);
  }

  neighbor_count_ = params_[kNeighborCount].GetInt32(0);
  filter_type_ = params_[kFilterType].GetInt32(0);

  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);

  if (filter_type_ > 0) {
    ADD_TENSOR(tensors_, kFilterIds, kInt64, kReservedSize);
    filter_ids_ = &(tensors_[kFilterIds]);
  }
}

void SamplingRequest::Set(const int64_t* src_ids,
                          int32_t batch_size) {
  src_ids_->AddInt64(src_ids, src_ids + batch_size);
}

void SamplingRequest::SetFilters(const int64_t* filter_ids,
                                 int32_t batch_size) {
  filter_ids_->AddInt64(filter_ids, filter_ids + batch_size);
}

void SamplingRequest::Set(const Tensor::Map& tensors) {
  const int64_t* src_ids = tensors.at(kSrcIds).GetInt64();
  int32_t batch_size = tensors.at(kSrcIds).Size();
  src_ids_->AddInt64(src_ids, src_ids + batch_size);

  if (filter_type_ > 0) {
    const int64_t* filter_ids = tensors.at(kFilterIds).GetInt64();
    batch_size = tensors.at(kFilterIds).Size();
    filter_ids_->AddInt64(filter_ids, filter_ids + batch_size);
  }
}

int32_t SamplingRequest::BatchSize() const {
  return src_ids_->Size();
}

const std::string& SamplingRequest::Type() const {
  return params_.at(kType).GetString(0);
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

const int64_t* SamplingRequest::GetFilters() const {
  if (filter_ids_) {
    return filter_ids_->GetInt64();
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

void SamplingResponse::Swap(OpResponse& right) {
  OpResponse::Swap(right);
  SamplingResponse& res = static_cast<SamplingResponse&>(right);
  std::swap(total_neighbor_count_, res.total_neighbor_count_);
  std::swap(neighbor_count_, res.neighbor_count_);
  std::swap(neighbors_, res.neighbors_);
  std::swap(edges_, res.edges_);
  std::swap(degrees_, res.degrees_);
}

void SamplingResponse::SerializeTo(void* response) {
  params_[kNeighborCount].SetInt32(1, total_neighbor_count_);
  OpResponse::SerializeTo(response);
}

void SamplingResponse::SetMembers() {
  auto cnt = &(params_[kNeighborCount]);
  if (cnt->Size() > 1) {
    neighbor_count_ = cnt->GetInt32(0);
    total_neighbor_count_ = cnt->GetInt32(1);
  }

  neighbors_ = &(tensors_[kNodeIds]);
  edges_ = &(tensors_[kEdgeIds]);
  if (tensors_.find(kDegreeKey) != tensors_.end()) {
    degrees_ = &(tensors_[kDegreeKey]);
  }
}

void SamplingResponse::Stitch(ShardsPtr<OpResponse> shards) {
  int32_t total_neighbor_count = 0;

  int32_t shard_id = 0;
  OpResponse* tmp = nullptr;
  while (shards->Next(&shard_id, &tmp)) {
    total_neighbor_count +=
      static_cast<SamplingResponse*>(tmp)->TotalNeighborCount();
  }

  shards->ResetNext();
  OpResponse::Stitch(shards);

  params_[kNeighborCount].SetInt32(1, total_neighbor_count);
  this->SetMembers();
}

void SamplingResponse::InitNeighborIds(int32_t count) {
  ADD_TENSOR(tensors_, kNodeIds, kInt64, count);
  neighbors_ = &(tensors_[kNodeIds]);
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
  params_[kNeighborCount].Resize(2);
  params_[kNeighborCount].SetInt32(0, neighbor_count);
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
REGISTER_SAMPING_REQUEST(RandomWithoutReplacement)
REGISTER_SAMPING_REQUEST(Topk)
REGISTER_SAMPING_REQUEST(EdgeWeight)
REGISTER_SAMPING_REQUEST(InDegree)
REGISTER_SAMPING_REQUEST(Full)
REGISTER_SAMPING_REQUEST(RandomNegative)
REGISTER_SAMPING_REQUEST(InDegreeNegative)
REGISTER_SAMPING_REQUEST(SoftInDegreeNegative)
REGISTER_SAMPING_REQUEST(NodeWeightNegative)

#undef REGISTER_SAMPING_REQUEST

}  // namespace graphlearn
