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

#include "include/op_request.h"

#include "core/partition/partitioner.h"
#include "include/constants.h"
#include "generated/proto/service.pb.h"

namespace graphlearn {

OpRequest::OpRequest(const std::string& shard_key)
    : ShardableRequest(shard_key),
      is_parse_from_(false) {
}

std::string OpRequest::Name() const {
  auto it = params_.find(kOpName);
  if (it != params_.end()) {
    return it->second.GetString(0);
  } else {
    return "OpRequest";
  }
}

OpRequest* OpRequest::Clone() const {
  OpRequest* req = new OpRequest(shard_key_);
  req->params_ = params_;
  return req;
}

void OpRequest::SerializeTo(void* request) {
  OpRequestPb* pb = static_cast<OpRequestPb*>(request);
  pb->set_name(Name());
  pb->set_shardable(shardable_);
  pb->set_need_server_ready(true);

  for (auto& param : params_) {
    Tensor* t = &(param.second);
    TensorValue* v = pb->add_params();
    v->set_name(param.first);
    v->set_length(t->Size());
    v->set_dtype(static_cast<int32_t>(t->DType()));
    t->SwapWithProto(v);
  }

  for (auto& tensor : tensors_) {
    Tensor* t = &(tensor.second);
    TensorValue* v = pb->add_tensors();
    v->set_name(tensor.first);
    v->set_length(t->Size());
    v->set_dtype(static_cast<int32_t>(t->DType()));
    t->SwapWithProto(v);
  }

  for (auto& tensor : sparse_tensors_) {
    SparseTensor* t = &(tensor.second);
    SparseTensorValue* v = pb->add_sparse_tensors();
    v->set_name(tensor.first);
    t->SwapWithProto(v);
  }

  is_parse_from_ = false;
}

bool OpRequest::ParseFrom(const void* request) {
  OpRequestPb* pb = const_cast<OpRequestPb*>(
    static_cast<const OpRequestPb*>(request));
  for (int32_t i = 0; i < pb->params_size(); ++i) {
    TensorValue* v = pb->mutable_params(i);
    DataType type = static_cast<DataType>(v->dtype());
    ADD_TENSOR(params_, v->name(), type, v->length());
    Tensor* t = &(params_[v->name()]);
    t->SwapWithProto(v);
  }

  for (int32_t i = 0; i < pb->tensors_size(); ++i) {
    TensorValue* v = pb->mutable_tensors(i);
    DataType type = static_cast<DataType>(v->dtype());
    ADD_TENSOR(tensors_, v->name(), type, v->length());
    Tensor* t = &(tensors_[v->name()]);
    t->SwapWithProto(v);
  }

  for (int32_t i = 0; i < pb->sparse_tensors_size(); ++i) {
    SparseTensorValue* v = pb->mutable_sparse_tensors(i);
    TensorValue* segments = v->mutable_segments();
    DataType type = static_cast<DataType>(segments->dtype());
    Tensor ind(type, segments->length());
    ind.SwapWithProto(segments);

    TensorValue* values = v->mutable_values();
    type = static_cast<DataType>(values->dtype());
    Tensor val(type, values->length());
    val.SwapWithProto(values);

    sparse_tensors_.emplace(v->name(),
      std::move(SparseTensor{std::move(ind), std::move(val)}));
  }

  shardable_ = pb->shardable();
  is_parse_from_ = true;
  this->Finalize();
  return true;
}

ShardsPtr<OpRequest> OpRequest::Partition() const {
  auto partitioner = GetPartitioner(this);
  return partitioner->Partition(this);
}

OpResponse::OpResponse()
    : batch_size_(0), is_parse_from_(false) {
}

void OpResponse::SerializeTo(void* response) {
  ADD_TENSOR(params_, kBatchSize, kInt32, 1);
  Tensor* bs  = &(params_[kBatchSize]);
  bs->Resize(1);
  bs->SetInt32(0, batch_size_);

  OpResponsePb* pb = static_cast<OpResponsePb*>(response);
  for (auto& param : params_) {
    Tensor* t = &(param.second);
    TensorValue* v = pb->add_params();
    v->set_name(param.first);
    v->set_length(t->Size());
    v->set_dtype(static_cast<int32_t>(t->DType()));
    t->SwapWithProto(v);
  }

  for (auto& tensor : tensors_) {
    Tensor* t = &(tensor.second);
    TensorValue* v = pb->add_tensors();
    v->set_name(tensor.first);
    v->set_length(t->Size());
    v->set_dtype(static_cast<int32_t>(t->DType()));
    t->SwapWithProto(v);
  }

  for (auto& tensor : sparse_tensors_) {
    SparseTensor* t = &(tensor.second);
    SparseTensorValue* v = pb->add_sparse_tensors();
    v->set_name(tensor.first);
    t->SwapWithProto(v);
  }

  is_parse_from_ = false;
}

bool OpResponse::ParseFrom(const void* response) {
  OpResponsePb* pb = const_cast<OpResponsePb*>(
    static_cast<const OpResponsePb*>(response));
  for (int32_t i = 0; i < pb->params_size(); ++i) {
    TensorValue* v = pb->mutable_params(i);
    DataType type = static_cast<DataType>(v->dtype());
    ADD_TENSOR(params_, v->name(), type, v->length());
    Tensor* t = &(params_[v->name()]);
    t->SwapWithProto(v);
  }

  for (int32_t i = 0; i < pb->tensors_size(); ++i) {
    TensorValue* v = pb->mutable_tensors(i);
    DataType type = static_cast<DataType>(v->dtype());
    ADD_TENSOR(tensors_, v->name(), type, v->length());
    Tensor* t = &(tensors_[v->name()]);
    t->SwapWithProto(v);
  }

  for (int32_t i = 0; i < pb->sparse_tensors_size(); ++i) {
    SparseTensorValue* v = pb->mutable_sparse_tensors(i);
    TensorValue* indices = v->mutable_segments();
    DataType type = static_cast<DataType>(indices->dtype());
    Tensor ind(type, indices->length());
    ind.SwapWithProto(indices);

    TensorValue* values = v->mutable_values();
    type = static_cast<DataType>(values->dtype());
    Tensor val(type, values->length());
    val.SwapWithProto(values);

    sparse_tensors_.emplace(v->name(),
      std::move(SparseTensor{std::move(ind), std::move(val)}));
  }

  batch_size_ = params_[kBatchSize].GetInt32(0);
  is_parse_from_ = true;
  this->Finalize();
  return true;
}

void OpResponse::Swap(OpResponse& right) {
  std::swap(batch_size_, right.batch_size_);
  std::swap(is_parse_from_, right.is_parse_from_);
  params_.swap(right.params_);
  tensors_.swap(right.tensors_);
  sparse_tensors_.swap(right.sparse_tensors_);
}

void OpResponse::Stitch(ShardsPtr<OpResponse> shards) {
  auto stitcher = GetStitcher(this);
  stitcher->Stitch(shards, this);
  this->Finalize();
}

void RequestFactory::Register(const std::string& name,
                              RequestCreator req_creator,
                              ResponseCreator res_creator) {
  std::unique_lock<std::mutex> _(mtx_);
  req_[name] = req_creator;
  res_[name] = res_creator;
}

OpRequest* RequestFactory::NewRequest(const std::string& name) {
  auto it = req_.find(name);
  if (it != req_.end()) {
    return it->second();
  }
  return nullptr;
}

OpResponse* RequestFactory::NewResponse(const std::string& name) {
  auto it = res_.find(name);
  if (it != res_.end()) {
    return it->second();
  }
  return nullptr;
}

}  // namespace graphlearn
