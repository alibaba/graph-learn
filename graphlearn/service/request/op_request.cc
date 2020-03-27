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

#include "graphlearn/include/op_request.h"

#include "graphlearn/core/partition/partitioner.h"
#include "graphlearn/include/constants.h"
#include "graphlearn/proto/service.pb.h"

namespace graphlearn {

namespace {

void SwapToPB(Tensor* t, TensorValue* v, DataType type) {
  if (type == DataType::kInt32) {
    t->SwapWithPB(v->mutable_int32_values());
  } else if (type == DataType::kInt64) {
    t->SwapWithPB(v->mutable_int64_values());
  } else if (type == DataType::kFloat) {
    t->SwapWithPB(v->mutable_float_values());
  } else if (type == DataType::kDouble) {
    t->SwapWithPB(v->mutable_double_values());
  } else if (type == DataType::kString) {
    // Because of the limitation of protobuf, Swap() does not work
    // well as expectattion. Here we use data copy instead. IF you
    // care the performance very much, please CONVERT the string
    // attributes into integer. We provide hash options for string
    // attributes of the Node and Edge. By doing this, the string
    // attributes will be converted into integers. Refer data_source.h
    // for details.
    for (int32_t i = 0; i < t->Size(); ++i) {
      v->add_string_values(t->GetString(i));
    }
  } else {
  }
}

void SwapFromPB(Tensor* t, TensorValue* v, DataType type) {
  if (type == DataType::kInt32) {
    t->SwapWithPB(v->mutable_int32_values());
  } else if (type == DataType::kInt64) {
    t->SwapWithPB(v->mutable_int64_values());
  } else if (type == DataType::kFloat) {
    t->SwapWithPB(v->mutable_float_values());
  } else if (type == DataType::kDouble) {
    t->SwapWithPB(v->mutable_double_values());
  } else if (type == DataType::kString) {
    // Because of the limitation of protobuf, Swap() does not work
    // well as expectattion. Here we use data copy instead. IF you
    // care the performance very much, please CONVERT the string
    // attributes into integer. We provide hash options for string
    // attributes of the Node and Edge. By doing this, the string
    // attributes will be converted into integers. Refer data_source.h
    // for details.
    for (int32_t i = 0; i < v->string_values_size(); ++i) {
      t->AddString(v->string_values(i));
    }
  } else {
  }
}

}  // anonymous namespace

OpRequest::OpRequest()
    : is_parse_from_(false) {
}

std::string OpRequest::Name() const {
  auto it = params_.find(kOpName);
  if (it != params_.end()) {
    return it->second.GetString(0);
  } else {
    return "OpRequest";
  }
}

bool OpRequest::HasPartitionKey() const {
  auto it = params_.find(kPartitionKey);
  return it != params_.end();
}

const std::string& OpRequest::PartitionKey() const {
  return params_.at(kPartitionKey).GetString(0);
}

OpRequest* OpRequest::Clone() const {
  OpRequest* req = new OpRequest;
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
    SwapToPB(t, v, t->DType());
  }

  for (auto& tensor : tensors_) {
    Tensor* t = &(tensor.second);
    TensorValue* v = pb->add_tensors();
    v->set_name(tensor.first);
    v->set_length(t->Size());
    v->set_dtype(static_cast<int32_t>(t->DType()));
    SwapToPB(t, v, t->DType());
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
    SwapFromPB(t, v, t->DType());
  }

  for (int32_t i = 0; i < pb->tensors_size(); ++i) {
    TensorValue* v = pb->mutable_tensors(i);
    DataType type = static_cast<DataType>(v->dtype());
    ADD_TENSOR(tensors_, v->name(), type, v->length());
    Tensor* t = &(tensors_[v->name()]);
    SwapFromPB(t, v, t->DType());
  }

  shardable_ = pb->shardable();
  is_parse_from_ = true;
  return true;
}

ShardsPtr<OpRequest> OpRequest::Partition() const {
  auto partitioner = GetPartitioner(this);
  return partitioner->Partition(this);
}

OpResponse::OpResponse()
    : batch_size_(0), is_sparse_(false), is_parse_from_(false) {
}

void OpResponse::SerializeTo(void* response) {
  ADD_TENSOR(params_, kBatchSize, kInt32, 2);
  Tensor* bs  = &(params_[kBatchSize]);
  bs->Resize(2);
  bs->SetInt32(0, batch_size_);
  bs->SetInt32(1, static_cast<int32_t>(is_sparse_));

  OpResponsePb* pb = static_cast<OpResponsePb*>(response);
  for (auto& param : params_) {
    Tensor* t = &(param.second);
    TensorValue* v = pb->add_params();
    v->set_name(param.first);
    v->set_length(t->Size());
    v->set_dtype(static_cast<int32_t>(t->DType()));
    SwapToPB(t, v, t->DType());
  }

  for (auto& tensor : tensors_) {
    Tensor* t = &(tensor.second);
    TensorValue* v = pb->add_tensors();
    v->set_name(tensor.first);
    v->set_length(t->Size());
    v->set_dtype(static_cast<int32_t>(t->DType()));
    SwapToPB(t, v, t->DType());
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
    SwapFromPB(t, v, t->DType());
  }

  for (int32_t i = 0; i < pb->tensors_size(); ++i) {
    TensorValue* v = pb->mutable_tensors(i);
    DataType type = static_cast<DataType>(v->dtype());
    ADD_TENSOR(tensors_, v->name(), type, v->length());
    Tensor* t = &(tensors_[v->name()]);
    SwapFromPB(t, v, t->DType());
  }

  batch_size_ = params_[kBatchSize].GetInt32(0);
  is_sparse_ = params_[kBatchSize].GetInt32(1) != 0;
  is_parse_from_ = true;
  return true;
}

void OpResponse::Swap(OpResponse& right) {
  std::swap(batch_size_, right.batch_size_);
  std::swap(is_sparse_, right.is_sparse_);
  std::swap(is_parse_from_, right.is_parse_from_);
  params_.swap(right.params_);
  tensors_.swap(right.tensors_);
}

void OpResponse::Stitch(ShardsPtr<OpResponse> shards) {
  auto stitcher = GetStitcher(this);
  stitcher->Stitch(shards, this);
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
