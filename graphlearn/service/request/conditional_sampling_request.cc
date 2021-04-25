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

ConditionalSamplingRequest::ConditionalSamplingRequest()
    : SamplingRequest(), dst_ids_(nullptr),
    int_cols_(nullptr), int_props_(nullptr),
    float_cols_(nullptr), float_props_(nullptr),
    str_cols_(nullptr), str_props_(nullptr) {
}

ConditionalSamplingRequest::ConditionalSamplingRequest(
  const std::string& type,
  const std::string& strategy,
  int32_t neighbor_count,
  const std::string& dst_node_type,
  bool batch_share,
  bool unique)
    : SamplingRequest(type,
                      "ConditionalNegativeSampler",
                      neighbor_count,
                      0),
      dst_ids_(nullptr),
      int_cols_(nullptr), int_props_(nullptr),
      float_cols_(nullptr), float_props_(nullptr),
      str_cols_(nullptr), str_props_(nullptr) {

  ADD_TENSOR(params_, kStrategy, kString, 1);
  params_[kStrategy].AddString(strategy);
  ADD_TENSOR(params_, kDstType, kString, 1);
  params_[kDstType].AddString(dst_node_type);
  ADD_TENSOR(params_, kBatchShare, kInt32, 1);
  params_[kBatchShare].AddInt32(batch_share ? 1 : 0);
  ADD_TENSOR(params_, kUnique, kInt32, 1);
  params_[kUnique].AddInt32(unique ? 1 : 0);

  ADD_TENSOR(params_, kIntCols, kInt32, 1);
  int_cols_ = &(params_[kIntCols]);
  ADD_TENSOR(params_, kIntProps, kFloat, 1);
  int_props_ = &(params_[kIntProps]);
  ADD_TENSOR(params_, kFloatCols, kInt32, 1);
  float_cols_ = &(params_[kFloatCols]);
  ADD_TENSOR(params_, kFloatProps, kFloat, 1);
  float_props_ = &(params_[kFloatProps]);
  ADD_TENSOR(params_, kStrCols, kInt32, 1);
  str_cols_ = &(params_[kStrCols]);
  ADD_TENSOR(params_, kStrProps, kFloat, 1);
  str_props_ = &(params_[kStrProps]);

  tensors_.reserve(2);
  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);
  ADD_TENSOR(tensors_, kDstIds, kInt64, kReservedSize);
  dst_ids_ = &(tensors_[kDstIds]);
}

OpRequest* ConditionalSamplingRequest::Clone() const {
  ConditionalSamplingRequest* req = new ConditionalSamplingRequest(
      Type(), Strategy(), neighbor_count_,
      DstNodeType(), BatchShare(), Unique());
  req->SetSelectedCols(IntCols(), IntProps(), FloatCols(), FloatProps(),
      StrCols(), StrProps());
  return req;
}

void ConditionalSamplingRequest::SetMembers() {
  neighbor_count_ = params_[kNeighborCount].GetInt32(0);
  int_cols_ = &(params_[kIntCols]);
  int_props_ = &(params_[kIntProps]);
  float_cols_ = &(params_[kFloatCols]);
  float_props_ = &(params_[kFloatProps]);
  str_cols_ = &(params_[kStrCols]);
  str_props_ = &(params_[kStrProps]);

  src_ids_ = &(tensors_[kSrcIds]);
  dst_ids_ = &(tensors_[kDstIds]);
}

void ConditionalSamplingRequest::Init(const Tensor::Map& params) {
  params_.reserve(kReservedSize);
  ADD_TENSOR(params_, kType, kString, 1);
  params_[kType].AddString(params.at(kEdgeType).GetString(0));
  ADD_TENSOR(params_, kPartitionKey, kString, 1);
  params_[kPartitionKey].AddString(kSrcIds);
  ADD_TENSOR(params_, kOpName, kString, 1);
  params_[kOpName].AddString("ConditionalNegativeSampler");
  ADD_TENSOR(params_, kStrategy, kString, 1);
  params_[kStrategy].AddString(params.at(kStrategy).GetString(0));
  ADD_TENSOR(params_, kNeighborCount, kInt32, 1);
  params_[kNeighborCount].AddInt32(params.at(kNeighborCount).GetInt32(0));
  ADD_TENSOR(params_, kDstType, kString, 1);
  params_[kDstType].AddString(params.at(kDstType).GetString(0));
  ADD_TENSOR(params_, kBatchShare, kInt32, 1);
  params_[kBatchShare].AddInt32(params.at(kBatchShare).GetInt32(0));
  ADD_TENSOR(params_, kUnique, kInt32, 1);
  params_[kUnique].AddInt32(params.at(kUnique).GetInt32(0));

  neighbor_count_ = params_[kNeighborCount].GetInt32(0);

  ADD_TENSOR(params_, kIntCols, kInt32, 1);
  int_cols_ = &(params_[kIntCols]);
  ADD_TENSOR(params_, kIntProps, kFloat, 1);
  int_props_ = &(params_[kIntProps]);
  ADD_TENSOR(params_, kFloatCols, kInt32, 1);
  float_cols_ = &(params_[kFloatCols]);
  ADD_TENSOR(params_, kFloatProps, kFloat, 1);
  float_props_ = &(params_[kFloatProps]);
  ADD_TENSOR(params_, kStrCols, kInt32, 1);
  str_cols_ = &(params_[kStrCols]);
  ADD_TENSOR(params_, kStrProps, kFloat, 1);
  str_props_ = &(params_[kStrProps]);

  tensors_.reserve(2);
  ADD_TENSOR(tensors_, kSrcIds, kInt64, kReservedSize);
  src_ids_ = &(tensors_[kSrcIds]);
  ADD_TENSOR(tensors_, kDstIds, kInt64, kReservedSize);
  dst_ids_ = &(tensors_[kDstIds]);

  // Set vector params
  if (params.find(kIntCols) != params.end()) {
    const int32_t* int_cols = params.at(kIntCols).GetInt32();
    int32_t int_cols_size = params.at(kIntCols).Size();
    int_cols_->AddInt32(int_cols, int_cols + int_cols_size);
  }

  if (params.find(kIntProps) != params.end()) {
    const float* int_props = params.at(kIntProps).GetFloat();
    int32_t int_props_size = params.at(kIntProps).Size();
    int_props_->AddFloat(int_props, int_props + int_props_size);
  }

  if (params.find(kFloatCols) != params.end()) {
    const int32_t* float_cols = params.at(kFloatCols).GetInt32();
    int32_t float_cols_size = params.at(kFloatCols).Size();
    float_cols_->AddInt32(float_cols, float_cols + float_cols_size);
  }

  if (params.find(kFloatProps) != params.end()) {
    const float* float_props = params.at(kFloatProps).GetFloat();
    int32_t float_props_size = params.at(kFloatProps).Size();
    float_props_->AddFloat(float_props, float_props + float_props_size);
  }

  if (params.find(kStrCols) != params.end()) {
    const int32_t* str_cols = params.at(kStrCols).GetInt32();
    int32_t str_cols_size = params.at(kStrCols).Size();
    str_cols_->AddInt32(str_cols, str_cols + str_cols_size);
  }

  if (params.find(kStrProps) != params.end()) {
    const float* str_props = params.at(kStrProps).GetFloat();
    int32_t str_props_size = params.at(kStrProps).Size();
    str_props_->AddFloat(str_props, str_props + str_props_size);
  }
}

void ConditionalSamplingRequest::Set(const Tensor::Map& tensors) {
  const int64_t* src_ids = tensors.at(kSrcIds).GetInt64();
  int32_t batch_size = tensors.at(kSrcIds).Size();
  src_ids_->AddInt64(src_ids, src_ids + batch_size);

  const int64_t* dst_ids = tensors.at(kDstIds).GetInt64();
  batch_size = tensors.at(kDstIds).Size();
  dst_ids_->AddInt64(dst_ids, dst_ids + batch_size);
}

void ConditionalSamplingRequest::SetIds(const int64_t* src_ids,
                                        const int64_t* dst_ids,
                                        int32_t batch_size) {
  src_ids_->AddInt64(src_ids, src_ids + batch_size);
  dst_ids_->AddInt64(dst_ids, dst_ids + batch_size);
}

void ConditionalSamplingRequest::SetSelectedCols(
    const std::vector<int32_t>& int_cols,
    const std::vector<float>& int_props,
    const std::vector<int32_t>& float_cols,
    const std::vector<float>& float_props,
    const std::vector<int32_t>& str_cols,
    const std::vector<float>& str_props) {
  int_cols_->AddInt32(int_cols.data(), int_cols.data() + int_cols.size());
  int_props_->AddFloat(int_props.data(), int_props.data() + int_props.size());
  float_cols_->AddInt32(float_cols.data(), float_cols.data() + float_cols.size());
  float_props_->AddFloat(float_props.data(), float_props.data() + float_props.size());
  str_cols_->AddInt32(str_cols.data(), str_cols.data() + str_cols.size());
  str_props_->AddFloat(str_props.data(), str_props.data() + str_props.size());
}

const std::string& ConditionalSamplingRequest::Strategy() const {
  return params_.at(kStrategy).GetString(0);
}

const std::string& ConditionalSamplingRequest::DstNodeType() const {
  return params_.at(kDstType).GetString(0);
}

const bool ConditionalSamplingRequest::BatchShare() const {
  return params_.at(kBatchShare).GetInt32(0) == 1;
}

const bool ConditionalSamplingRequest::Unique() const {
  return params_.at(kUnique).GetInt32(0) == 1;
}

const int64_t* ConditionalSamplingRequest::GetDstIds() const {
  if (dst_ids_) {
    return dst_ids_->GetInt64();
  } else {
    return nullptr;
  }
}

const std::vector<int32_t> ConditionalSamplingRequest::IntCols() const {
  if (int_cols_) {
    return std::vector<int32_t>(int_cols_->GetInt32(),
                                int_cols_->GetInt32() + int_cols_->Size());
  } else {
    return std::vector<int32_t>();
  }
}

const std::vector<float> ConditionalSamplingRequest::IntProps() const {
  if (int_props_) {
    return std::vector<float>(int_props_->GetFloat(),
                              int_props_->GetFloat() + int_props_->Size());
  } else {
    return std::vector<float>();
  }
}
const std::vector<int32_t> ConditionalSamplingRequest::FloatCols() const {
  if (float_cols_) {
    return std::vector<int32_t>(float_cols_->GetInt32(),
                                float_cols_->GetInt32() + float_cols_->Size());
  } else {
    return std::vector<int32_t>();
  }
}

const std::vector<float> ConditionalSamplingRequest::FloatProps() const {
  if (float_props_) {
    return std::vector<float>(float_props_->GetFloat(),
                              float_props_->GetFloat() + float_props_->Size());
  } else {
    return std::vector<float>();
  }
}

const std::vector<int32_t> ConditionalSamplingRequest::StrCols() const {
  if (str_cols_) {
    return std::vector<int32_t>(str_cols_->GetInt32(),
                                str_cols_->GetInt32() + str_cols_->Size());
  } else {
    return std::vector<int32_t>();
  }
}

const std::vector<float> ConditionalSamplingRequest::StrProps() const {
  if (str_props_) {
    return std::vector<float>(str_props_->GetFloat(),
                              str_props_->GetFloat() + str_props_->Size());
  } else {
    return std::vector<float>();
  }
}

REGISTER_REQUEST(ConditionalNegativeSampler,
                 ConditionalSamplingRequest,
                 SamplingResponse)

}  // namespace graphlearn
