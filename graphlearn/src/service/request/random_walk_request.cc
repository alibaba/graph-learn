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

#include "include/random_walk_request.h"

#include <cmath>
#include <float.h>
#include "include/constants.h"

namespace graphlearn {

namespace {
  int32_t kReservedSize = 64;
}  // anonymous namespace

RandomWalkRequest::RandomWalkRequest()
    : OpRequest(kSrcIds),
      src_ids_(nullptr),
      parent_ids_(nullptr),
      parent_neighbor_ids_(nullptr),
      parent_neighbor_segments_(nullptr) {
}

RandomWalkRequest::RandomWalkRequest(const std::string& type,
                                     float p,
                                     float q,
                                     int32_t walk_len)
    : OpRequest(kSrcIds),
      src_ids_(nullptr),
      parent_ids_(nullptr),
      parent_neighbor_ids_(nullptr),
      parent_neighbor_segments_(nullptr) {
  params_.reserve(kReservedSize);
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("RandomWalk");

  ADD_TENSOR(params_, kEdgeType, kString, 1);
  params_[kEdgeType].AddString(type);

  ADD_TENSOR(params_, kSideInfo, kFloat, 2);
  params_[kSideInfo].AddFloat(p);
  params_[kSideInfo].AddFloat(q);

  ADD_TENSOR(params_, kDistances, kInt32, 1);
  params_[kDistances].AddInt32(walk_len);

  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);  // size = batch_size

  if (!IsDeepWalk()) {
    ADD_TENSOR(tensors_, kNodeIds, kInt64, kReservedSize);
    parent_ids_ = &(tensors_[kNodeIds]); // size  = batch size

    Tensor parent_nbr_ids(kInt64, kReservedSize);
    Tensor parent_nbr_indices(kInt32, kReservedSize);
    sparse_tensors_.emplace(kSparseIds,
      std::move(SparseTensor{std::move(parent_nbr_indices), std::move(parent_nbr_ids)}));
    parent_neighbor_segments_ = sparse_tensors_[kSparseIds].MutableSegments();
    parent_neighbor_ids_ = sparse_tensors_[kSparseIds].MutableValues();
  }
}

OpRequest* RandomWalkRequest::Clone() const {
  RandomWalkRequest* req = new RandomWalkRequest(
    Type(), P(), Q(), WalkLen());
  return req;
}

void RandomWalkRequest::Finalize() {
  src_ids_ = &(tensors_[kSrcIds]);
  if (!IsDeepWalk()) {
    parent_ids_ = &(tensors_[kNodeIds]);
    parent_neighbor_segments_ = sparse_tensors_[kSparseIds].MutableSegments();
    parent_neighbor_ids_ = sparse_tensors_[kSparseIds].MutableValues();
  }
}

void RandomWalkRequest::Init(const Tensor::Map& params) {
  params_.reserve(kReservedSize);
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("RandomWalk");

  ADD_TENSOR(params_, kEdgeType, kString, 1);
  params_[kEdgeType].AddString(params.at(kEdgeType).GetString(0));

  ADD_TENSOR(params_, kSideInfo, kFloat, 2);
  params_[kSideInfo].AddFloat(params.at(kSideInfo).GetFloat(0));
  params_[kSideInfo].AddFloat(params.at(kSideInfo).GetFloat(1));

  ADD_TENSOR(params_, kDistances, kInt32, 1);
  params_[kDistances].AddInt32(params.at(kDistances).GetInt32(0));

  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);

  if (!IsDeepWalk()) {
    ADD_TENSOR(tensors_, kNodeIds, kInt64, kReservedSize);
    parent_ids_ = &(tensors_[kNodeIds]);

    Tensor parent_nbr_ids(kInt64, kReservedSize);
    Tensor parent_nbr_indices(kInt32, kReservedSize);
    sparse_tensors_.emplace(kSparseIds,
      std::move(SparseTensor{std::move(parent_nbr_indices), std::move(parent_nbr_ids)}));
    parent_neighbor_segments_ = sparse_tensors_[kSparseIds].MutableSegments();
    parent_neighbor_ids_ = sparse_tensors_[kSparseIds].MutableValues();
  }
}

void RandomWalkRequest::Set(const Tensor::Map& tensors, const SparseTensor::Map& sparse_tensors) {
  auto src_ids = tensors.at(kSrcIds).GetInt64();
  auto batch_size = tensors.at(kSrcIds).Size();
  src_ids_->AddInt64(src_ids, src_ids + batch_size);

  if (!IsDeepWalk()) {
    parent_ids_->AddInt64(src_ids, src_ids + batch_size);
    for (int32_t i = 0; i < batch_size; ++i) {
      parent_neighbor_segments_->AddInt32(0);
    }
  }
}

void RandomWalkRequest::Set(const int64_t* src_ids,
                            int32_t batch_size) {
  src_ids_->AddInt64(src_ids, src_ids + batch_size);
}

void RandomWalkRequest::Set(const int64_t* src_ids,
                            const int64_t* parent_ids,
                            int32_t batch_size,
                            const int64_t* parent_neighbor_ids,
                            const int32_t* parent_neighbor_segments,
                            int32_t total_count) {
  src_ids_->AddInt64(src_ids, src_ids + batch_size);
  parent_ids_->AddInt64(parent_ids, parent_ids + batch_size);
  parent_neighbor_ids_->AddInt64(parent_neighbor_ids, parent_neighbor_ids + total_count);
  parent_neighbor_segments_->AddInt32(parent_neighbor_segments, parent_neighbor_segments + batch_size);
}

int32_t RandomWalkRequest::BatchSize() const {
  return src_ids_->Size();
}

const bool RandomWalkRequest::IsDeepWalk() const {
  auto p = params_.at(kSideInfo).GetFloat(0);
  auto q = params_.at(kSideInfo).GetFloat(1);
  if (fabs(p - float(1.0)) < 32 * FLT_EPSILON &&
      fabs(q - float(1.0)) < 32 * FLT_EPSILON) {
    return true;
  }
  return false;
}

const std::string& RandomWalkRequest::Type() const {
  return params_.at(kEdgeType).GetString(0);
}

const float RandomWalkRequest::P() const {
  return params_.at(kSideInfo).GetFloat(0);
}

const float RandomWalkRequest::Q() const {
  return params_.at(kSideInfo).GetFloat(1);
}

const int32_t RandomWalkRequest::WalkLen() const {
  return params_.at(kDistances).GetInt32(0);
}

const int64_t* RandomWalkRequest::GetSrcIds() const {
  if (src_ids_) {
    return src_ids_->GetInt64();
  } else {
    return nullptr;
  }
}

const int64_t* RandomWalkRequest::GetParentIds() const {
  if (parent_ids_) {
    return parent_ids_->GetInt64();
  } else {
    return nullptr;
  }
}

const int64_t* RandomWalkRequest::GetParentNeighborIds() const {
  if (parent_neighbor_ids_) {
    return parent_neighbor_ids_->GetInt64();
  } else {
    return nullptr;
  }
}

const int32_t* RandomWalkRequest::GetParentNeighborSegments() const {
  if (parent_neighbor_segments_) {
    return parent_neighbor_segments_->GetInt32();
  } else {
    return nullptr;
  }
}

RandomWalkResponse::RandomWalkResponse()
    : OpResponse(),
      ids_(nullptr),
      neighbors_(nullptr),
      degrees_(nullptr) {
}

void RandomWalkResponse::Swap(OpResponse& right) {
  OpResponse::Swap(right);
  RandomWalkResponse& res = static_cast<RandomWalkResponse&>(right);
  std::swap(ids_, res.ids_);
  std::swap(neighbors_, res.neighbors_);
  std::swap(degrees_, res.degrees_);
}

void RandomWalkResponse::Finalize() {
  ids_ = &(tensors_[kNodeIds]);

  if (sparse_tensors_.find(kDstIds) != sparse_tensors_.end()) {
    neighbors_ = sparse_tensors_[kDstIds].MutableValues();
    degrees_ = sparse_tensors_[kDstIds].MutableSegments();
  }
}

void RandomWalkResponse::InitWalks(int32_t count) {
  ADD_TENSOR(tensors_, kNodeIds, kInt64, count);
  ids_ = &(tensors_[kNodeIds]);
}

void RandomWalkResponse::InitNeighbors(int32_t batch_size,
                                       int32_t total_count) {
  Tensor indices(kInt32, batch_size);
  Tensor values(kInt64, total_count);
  sparse_tensors_.emplace(kDstIds,
    std::move(SparseTensor{std::move(indices), std::move(values)}));
  neighbors_ = sparse_tensors_[kDstIds].MutableValues();
  degrees_ = sparse_tensors_[kDstIds].MutableSegments();
}

void RandomWalkResponse::SetBatchSize(int32_t batch_size) {
  batch_size_ = batch_size;
}

void RandomWalkResponse::AppendWalks(const int64_t* walks, size_t size) {
  ids_->AddInt64(walks, walks + size);
}

void RandomWalkResponse::AppendNeighborIds(const int64_t* ids, size_t size) {
  neighbors_->AddInt64(ids, ids + size);
}

void RandomWalkResponse::AppendDegrees(const int32_t* degrees, size_t size) {
  degrees_->AddInt32(degrees, degrees + size);
}

const int64_t* RandomWalkResponse:: GetWalks() const {
  if (ids_) {
    return ids_->GetInt64();
  } else {
    return nullptr;
  }
}

const int64_t* RandomWalkResponse::GetNeighborIds() const {
  if (neighbors_) {
    return neighbors_->GetInt64();
  } else {
    return nullptr;
  }
}

const int32_t* RandomWalkResponse::GetDegrees() const {
  if (degrees_) {
    return degrees_->GetInt32();
  } else {
    return nullptr;
  }
}

REGISTER_REQUEST(RandomWalk, RandomWalkRequest, RandomWalkResponse);

}  // namespace graphlearn
