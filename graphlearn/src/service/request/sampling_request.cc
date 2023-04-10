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

#include "include/sampling_request.h"

#include "include/constants.h"

namespace graphlearn {

namespace {
int32_t kReservedSize = 64;
}  // anonymous namespace

SamplingRequest::SamplingRequest()
  : OpRequest(kSrcIds),
    neighbor_count_(0),
    src_ids_(nullptr),
    filter_() {
}

SamplingRequest::SamplingRequest(const std::string& type,
                                 const std::string& strategy,
                                 int32_t neighbor_count,
                                 FilterType filter_type,
                                 FilterField filter_field)
    : OpRequest(kSrcIds),
      neighbor_count_(neighbor_count),
      src_ids_(nullptr),
      filter_(filter_field, filter_type) {
  params_.reserve(kReservedSize);

  ADD_TENSOR(params_, kType, kString, 1);
  params_[kType].AddString(type);

  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(strategy);

  ADD_TENSOR(params_, kNeighborCount, kInt32, 1);
  params_[kNeighborCount].AddInt32(neighbor_count);

  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);

  ADD_TENSOR(params_, kFilterType, kInt32, 1);
  params_[kFilterType].AddInt32(filter_type);

  ADD_TENSOR(params_, kFilterField, kInt32, 1);
  params_[kFilterField].AddInt32(filter_field);

  if (filter_) {
    ADD_TENSOR(tensors_, kFilterValues, kInt64, kReservedSize);
    filter_.InitValues(&tensors_[kFilterValues]);
  }
}

OpRequest* SamplingRequest::Clone() const {
  SamplingRequest* req = new SamplingRequest(
    Type(), Strategy(), neighbor_count_, filter_.GetType(), filter_.GetField());
  return req;
}

void SamplingRequest::Finalize() {
  neighbor_count_ = params_[kNeighborCount].GetInt32(0);
  src_ids_ = &(tensors_[kSrcIds]);

  // Set for filters
  op::Filter filter(static_cast<FilterField>(params_[kFilterField].GetInt32(0)),
                    static_cast<FilterType>(params_[kFilterType].GetInt32(0)));
  filter_ = std::move(filter);
  if (filter_) {
    filter_.InitValues(&(tensors_[kFilterValues]));
  }
}

void SamplingRequest::Init(const Tensor::Map& params) {
  params_.reserve(kReservedSize);
  ADD_TENSOR(params_, kType, kString, 1);
  params_[kType].AddString(params.at(kEdgeType).GetString(0));
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(params.at(kStrategy).GetString(0));
  ADD_TENSOR(params_, kNeighborCount, kInt32, 1);
  params_[kNeighborCount].AddInt32(params.at(kNeighborCount).GetInt32(0));

  neighbor_count_ = params_[kNeighborCount].GetInt32(0);
  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);

  ADD_TENSOR(params_, kFilterType, kInt32, 1);
  if (params.find(kFilterType) != params.end()) {
    params_[kFilterType].AddInt32(params.at(kFilterType).GetInt32(0));
  } else {
    params_[kFilterType].AddInt32(0);
  }
  ADD_TENSOR(params_, kFilterField, kInt32, 1);
  if (params.find(kFilterField) != params.end()) {
    params_[kFilterField].AddInt32(params.at(kFilterField).GetInt32(0));
  } else {
    params_[kFilterField].AddInt32(0);
  }

  op::Filter filter(static_cast<FilterField>(params_[kFilterField].GetInt32(0)),
                    static_cast<FilterType>(params_[kFilterType].GetInt32(0)));
  filter_ = std::move(filter);

  if (filter_) {
    ADD_TENSOR(tensors_, kFilterValues, kInt64, kReservedSize);
    filter_.InitValues(&(tensors_[kFilterValues]));
  }
}

void SamplingRequest::Set(const int64_t* src_ids,
                          int32_t batch_size) {
  src_ids_->AddInt64(src_ids, src_ids + batch_size);
}

void SamplingRequest::Set(const Tensor::Map& tensors, const SparseTensor::Map& sparse_tensors) {
  const int64_t* src_ids = tensors.at(kSrcIds).GetInt64();
  int32_t batch_size = tensors.at(kSrcIds).Size();
  src_ids_->AddInt64(src_ids, src_ids + batch_size);

  if (filter_) {
    filter_.FillValues(tensors.at(kFilterValues), batch_size);
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

const op::Filter* SamplingRequest::GetFilter() const {
  return &filter_;
}

SamplingResponse::SamplingResponse()
    : OpResponse(),
      shape_(),
      neighbors_(nullptr),
      edges_(nullptr) {
}

void SamplingResponse::Swap(OpResponse& right) {
  OpResponse::Swap(right);
  SamplingResponse& res = static_cast<SamplingResponse&>(right);
  shape_.Swap(res.shape_);
  std::swap(neighbors_, res.neighbors_);
  std::swap(edges_, res.edges_);
}

void SamplingResponse::Finalize() {
  auto dim2 = params_[kNeighborCount].GetInt32(0);
  auto iter = tensors_.find(kNodeIds);
  if (iter != tensors_.end()) {
    neighbors_ = &(iter->second);
    shape_ = Shape(batch_size_, dim2);
  } else {
    neighbors_ = sparse_tensors_[kNodeIds].MutableValues();
    auto& segment = sparse_tensors_[kNodeIds].Segments();
    shape_ = Shape(batch_size_, dim2,
       std::vector<int32_t>(segment.GetInt32(), segment.GetInt32() + segment.Size()));
  }

  iter = tensors_.find(kEdgeIds);
  if (iter != tensors_.end()) {
    edges_ = &(iter->second);
  } else {
    edges_ = sparse_tensors_[kEdgeIds].MutableValues();
  }
}

void SamplingResponse::InitNeighborIds() {
  if (!shape_.sparse) {
    Tensor values(kInt64, shape_.size);
    tensors_.emplace(kNodeIds, std::move(values));
    neighbors_ = &(tensors_[kNodeIds]);
  } else {
    Tensor values(kInt64, shape_.size);
    Tensor segments(kInt32, shape_.dim1);
    segments.AddInt32(shape_.segments.data(), shape_.segments.data() + shape_.segments.size());
    sparse_tensors_.emplace(kNodeIds, std::move(SparseTensor{std::move(segments), std::move(values)}));
    neighbors_ = sparse_tensors_[kNodeIds].MutableValues();
  }
}

void SamplingResponse::InitEdgeIds() {
  if (!shape_.sparse) {
    Tensor values(kInt64, shape_.size);
    tensors_.emplace(kEdgeIds, std::move(values));
    edges_ = &(tensors_[kEdgeIds]);
  } else {
    Tensor values(kInt64, shape_.size);
    Tensor segments(kInt32, shape_.dim1);
    segments.AddInt32(shape_.segments.data(), shape_.segments.data() + shape_.segments.size());
    sparse_tensors_.emplace(kEdgeIds, std::move(SparseTensor{std::move(segments), std::move(values)}));
    edges_ = sparse_tensors_[kEdgeIds].MutableValues();
  }
}

void SamplingResponse::SetShape(size_t dim1, size_t dim2) {
  // batch_size_ and neighbor_count are stored in params_
  batch_size_ = dim1;

  ADD_TENSOR(params_, kNeighborCount, kInt32, 1);
  params_[kNeighborCount].AddInt32(dim2);

  shape_ = Shape(dim1, dim2);
}

void SamplingResponse::SetShape(size_t dim1, size_t dim2, const std::vector<int32_t>& segments) {
  // batch_size_ and neighbor_count are stored in params_
  batch_size_ = dim1;

  ADD_TENSOR(params_, kNeighborCount, kInt32, 1);
  params_[kNeighborCount].AddInt32(dim2);

  shape_ = Shape(dim1, dim2, segments);
}

void SamplingResponse::SetShape(size_t dim1, size_t dim2, std::vector<int32_t>&& segments) {
  // batch_size_ and neighbor_count are stored in params_
  batch_size_ = dim1;

  ADD_TENSOR(params_, kNeighborCount, kInt32, 1);
  params_[kNeighborCount].AddInt32(dim2);

  shape_ = Shape(dim1, dim2, segments);
}

void SamplingResponse::AppendNeighborId(int64_t id) {
  neighbors_->AddInt64(id);
}

void SamplingResponse::AppendEdgeId(int64_t id) {
  edges_->AddInt64(id);
}

void SamplingResponse::FillWith(int64_t neighbor_id, int64_t edge_id) {
  for (int32_t i = 0; i < shape_.dim2; ++i) {
    neighbors_->AddInt64(neighbor_id);
  }

  // edges is optional
  if (edges_ != nullptr) {
    for (int32_t i = 0; i < shape_.dim2; ++i) {
      edges_->AddInt64(edge_id);
    }
  }
}

const Shape SamplingResponse::GetShape() const {
  return shape_;
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
