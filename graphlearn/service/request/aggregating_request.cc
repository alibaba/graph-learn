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

#include "graphlearn/include/aggregating_request.h"

#include "graphlearn/include/constants.h"
#include "graphlearn/core/operator/operator.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/core/operator/aggregator/aggregator.h"

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
  AggregatingRequest* req =
      new AggregatingRequest(Type(), Strategy());
  req->num_segments_ = num_segments_;
  return req;
}

void AggregatingRequest::SerializeTo(void* request) {
  ADD_TENSOR(params_, kNumSegments, kInt32, 1);
  params_[kNumSegments].AddInt32(num_segments_);
  OpRequest::SerializeTo(request);
}

bool AggregatingRequest::ParseFrom(const void* request) {
  if (!OpRequest::ParseFrom(request)) {
    return false;
  }
  num_segments_ = params_[kNumSegments].GetInt32(0);
  node_ids_ = &(tensors_[kNodeIds]);
  segment_ids_ = &(tensors_[kSegmentIds]);
  return true;
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

bool AggregatingResponse::ParseFrom(const void* response) {
  if (!OpResponse::ParseFrom(response)) {
    return false;
  }

  embs_ = &(tensors_[kFloatAttrKey]);
  segments_ = &(tensors_[kSegments]);
  emb_dim_ = params_[kSideInfo].GetInt32(0);
  name_ = params_[kOpName].GetString(0);
  return true;
}

void AggregatingResponse::Stitch(ShardsPtr<OpResponse> shards) {
  int32_t shard_id = 0;
  OpResponse* tmp = nullptr;
  shards->Next(&shard_id, &tmp);

  AggregatingResponse* agg_tmp = nullptr;
  agg_tmp = static_cast<AggregatingResponse*>(tmp);
  int32_t num_segments = agg_tmp->NumSegments();
  int32_t emb_dim = agg_tmp->EmbeddingDim();
  int32_t size = num_segments * emb_dim;
  const std::string& name = agg_tmp->Name();

  op::Operator* op = op::OperatorFactory::GetInstance().Lookup(name);
  op::Aggregator* agg_op = static_cast<op::Aggregator*>(op);

  tensors_.clear();
  tensors_.reserve(2);

  ADD_TENSOR(tensors_, kFloatAttrKey, kFloat, size);
  tensors_[kFloatAttrKey].Resize(size);
  embs_ = &(tensors_[kFloatAttrKey]);
  float* embs = const_cast<float*>(embs_->GetFloat());

  ADD_TENSOR(tensors_, kSegments, kInt32, num_segments);
  tensors_[kSegments].Resize(num_segments);
  segments_ = &(tensors_[kSegments]);
  int32_t* segments = const_cast<int32_t*>(segments_->GetInt32());

  agg_op->InitFunc(embs, size);

  shards->ResetNext();
  while (shards->Next(&shard_id, &tmp)) {
    agg_tmp = static_cast<AggregatingResponse*>(tmp);
    float* tmp_embs = const_cast<float*>(agg_tmp->Embeddings());
    const int32_t* tmp_segments = agg_tmp->Segments();
    agg_op->AggFunc(embs, tmp_embs, size, tmp_segments, num_segments);
    for (int32_t i = 0; i < num_segments; ++i) {
      segments[i] += tmp_segments[i];
    }
  }

  agg_op->FinalFunc(embs, size, segments, num_segments);

  batch_size_ = num_segments;

  name_ = name;
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString(name_);

  emb_dim_ = emb_dim;
  ADD_TENSOR(params_, kSideInfo, kInt32, 1);
  params_[kSideInfo].AddInt32(emb_dim_);
}

REGISTER_REQUEST(MinAggregator, AggregatingRequest, AggregatingResponse);
REGISTER_REQUEST(ProdAggregator, AggregatingRequest, AggregatingResponse);
REGISTER_REQUEST(SumAggregator, AggregatingRequest, AggregatingResponse);
REGISTER_REQUEST(MaxAggregator, AggregatingRequest, AggregatingResponse);
REGISTER_REQUEST(MeanAggregator, AggregatingRequest, AggregatingResponse);

}  // namespace graphlearn
