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

#include "include/aggregating_request.h"

#include "include/constants.h"
#include "core/operator/operator.h"
#include "core/operator/op_factory.h"
#include "core/operator/aggregator/aggregator.h"

namespace graphlearn {

namespace {
int32_t kReservedSize = 64;
}  // anonymous namespace

AggregatingRequest::AggregatingRequest()
    : OpRequest(), cursor_(0), num_segments_(0),
      node_ids_(nullptr), segment_ids_(nullptr) {
}

AggregatingRequest::AggregatingRequest(const std::string& type,
                                       const std::string& strategy)
    : OpRequest(), cursor_(0), num_segments_(0),
      node_ids_(nullptr), segment_ids_(nullptr) {
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(strategy);

  ADD_TENSOR(params_, kPartitionKey, kString, 1);
  params_[kPartitionKey].AddString(kNodeIds);

  ADD_TENSOR(params_, kNodeType, kString, 1);
  params_[kNodeType].AddString(type);

  ADD_TENSOR(tensors_, kNodeIds, kInt64, kReservedSize);
  node_ids_ = &(tensors_[kNodeIds]);

  ADD_TENSOR(tensors_, kSegmentIds, kInt32, kReservedSize);
  segment_ids_ = &(tensors_[kSegmentIds]);
}

OpRequest* AggregatingRequest::Clone() const {
  AggregatingRequest* req = new AggregatingRequest(Type(), Strategy());
  req->num_segments_ = num_segments_;
  return req;
}

void AggregatingRequest::SerializeTo(void* request) {
  ADD_TENSOR(params_, kNumSegments, kInt32, 1);
  params_[kNumSegments].AddInt32(num_segments_);
  OpRequest::SerializeTo(request);
}

void AggregatingRequest::SetMembers() {
  num_segments_ = params_[kNumSegments].GetInt32(0);
  node_ids_ = &(tensors_[kNodeIds]);
  segment_ids_ = &(tensors_[kSegmentIds]);
}

void AggregatingRequest::Set(const int64_t* node_ids,
                             const int32_t* segment_ids,
                             int32_t num_ids,
                             int32_t num_segments) {
  node_ids_->AddInt64(node_ids, node_ids + num_ids);
  segment_ids_->AddInt32(segment_ids, segment_ids + num_ids);
  num_segments_ = num_segments;
}

const std::string& AggregatingRequest::Type() const {
  return params_.at(kNodeType).GetString(0);
}

const std::string& AggregatingRequest::Strategy() const {
  return params_.at(kOpName).GetString(0);
}

bool AggregatingRequest::Next(int64_t* node_id, int32_t* segment_id) {
  if (cursor_ >= NumIds()) {
    return false;
  }

  *node_id = node_ids_->GetInt64(cursor_);
  *segment_id = segment_ids_->GetInt32(cursor_);
  ++cursor_;
  return true;
}

bool AggregatingRequest::SegmentEnd(int32_t segment_id) const {
  if (cursor_ >= NumIds()) {
    return true;
  }
  if (segment_ids_->GetInt32(cursor_) != segment_id) {
    return true;
  }
  return false;
}

AggregatingResponse::AggregatingResponse()
    : OpResponse(),
      name_(""),
      emb_dim_(0),
      embs_(nullptr),
      segments_(nullptr) {
}

void AggregatingResponse::Swap(OpResponse& right) {
  OpResponse::Swap(right);
  AggregatingResponse& res = static_cast<AggregatingResponse&>(right);
  std::swap(name_, res.name_);
  std::swap(emb_dim_, res.emb_dim_);
  std::swap(embs_, res.embs_);
  std::swap(segments_, res.segments_);
}

void AggregatingResponse::SetName(const std::string& name) {
  name_ = name;
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(name_);

  ADD_TENSOR(tensors_, kFloatAttrKey, kFloat, kReservedSize);
  embs_ = &(tensors_[kFloatAttrKey]);

  ADD_TENSOR(tensors_, kSegments, kInt32, kReservedSize);
  segments_ = &(tensors_[kSegments]);
}

void AggregatingResponse::SetEmbeddingDim(int32_t dim) {
  emb_dim_ = dim;
  ADD_TENSOR(params_, kSideInfo, kInt32, 1);
  params_[kSideInfo].AddInt32(emb_dim_);
}

void AggregatingResponse::SetNumSegments(int32_t num_segments) {
  // the number of segments is batch_size
  batch_size_ = num_segments;
}

void AggregatingResponse::AppendEmbedding(const float* value) {
  for (int32_t i = 0; i < emb_dim_; ++i) {
    embs_->AddFloat(value[i]);
  }
}

const float* AggregatingResponse::Embeddings() const {
  return embs_->GetFloat();
}

void AggregatingResponse::AppendSegment(int32_t size) {
  segments_->AddInt32(size);
}

const int32_t* AggregatingResponse::Segments() const {
  return segments_->GetInt32();
}

void AggregatingResponse::SetMembers() {
  embs_ = &(tensors_[kFloatAttrKey]);
  segments_ = &(tensors_[kSegments]);
  emb_dim_ = params_[kSideInfo].GetInt32(0);
  name_ = params_[kOpName].GetString(0);
}

void AggregatingResponse::Stitch(ShardsPtr<OpResponse> shards) {
  int32_t shard_id = 0;
  OpResponse* tmp = nullptr;
  shards->Next(&shard_id, &tmp);
  AggregatingResponse* shard = static_cast<AggregatingResponse*>(tmp);

  batch_size_ = shard->NumSegments();
  int32_t emb_dim = shard->EmbeddingDim();
  int32_t size = batch_size_ * emb_dim;

  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(shard->Name());
  ADD_TENSOR(params_, kSideInfo, kInt32, 1);
  params_[kSideInfo].AddInt32(emb_dim);

  tensors_.clear();
  tensors_.reserve(2);
  ADD_TENSOR(tensors_, kFloatAttrKey, kFloat, size);
  tensors_[kFloatAttrKey].Resize(size);
  ADD_TENSOR(tensors_, kSegments, kInt32, batch_size_);
  tensors_[kSegments].Resize(batch_size_);

  float* embs = const_cast<float*>(tensors_[kFloatAttrKey].GetFloat());
  int32_t* segments = const_cast<int32_t*>(tensors_[kSegments].GetInt32());

  op::Operator* op = op::OpFactory::GetInstance()->Create(shard->Name());
  op::Aggregator* agg_op = static_cast<op::Aggregator*>(op);
  agg_op->InitFunc(embs, size);

  shards->ResetNext();
  while (shards->Next(&shard_id, &tmp)) {
    shard = static_cast<AggregatingResponse*>(tmp);
    float* shard_embs = const_cast<float*>(shard->Embeddings());
    const int32_t* shard_segments = shard->Segments();
    agg_op->AggFunc(embs, shard_embs, size, shard_segments, batch_size_);
    for (int32_t i = 0; i < batch_size_; ++i) {
      segments[i] += shard_segments[i];
    }
  }
  agg_op->FinalFunc(embs, size, segments, batch_size_);
  this->SetMembers();
}

REGISTER_REQUEST(MinAggregator, AggregatingRequest, AggregatingResponse);
REGISTER_REQUEST(ProdAggregator, AggregatingRequest, AggregatingResponse);
REGISTER_REQUEST(SumAggregator, AggregatingRequest, AggregatingResponse);
REGISTER_REQUEST(MaxAggregator, AggregatingRequest, AggregatingResponse);
REGISTER_REQUEST(MeanAggregator, AggregatingRequest, AggregatingResponse);

}  // namespace graphlearn
