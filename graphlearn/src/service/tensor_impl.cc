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

#include "service/tensor_impl.h"

#include "common/base/log.h"

namespace graphlearn {

TensorImpl::TensorImpl(DataType dtype)
    : type_(dtype),
      size_(0),
      int32_buf_(nullptr),
      int64_buf_(nullptr),
      float_buf_(nullptr),
      double_buf_(nullptr),
      string_buf_(nullptr) {
  if (type_ == DataType::kInt32) {
    int32_buf_ = new ::PB_NAMESPACE::RepeatedField<int32_t>();
  } else if (type_ == DataType::kInt64) {
    int64_buf_ = new ::PB_NAMESPACE::RepeatedField<int64_t>();
  } else if (type_ == DataType::kFloat) {
    float_buf_ = new ::PB_NAMESPACE::RepeatedField<float>();
  } else if (type_ == DataType::kDouble) {
    double_buf_ = new ::PB_NAMESPACE::RepeatedField<double>();
  } else if (type_ == DataType::kString) {
    string_buf_ = new ::PB_NAMESPACE::RepeatedPtrField<std::string>();
  } else {
    LOG(ERROR) << "Invalid data type: " << static_cast<int32_t>(dtype);
  }
}

TensorImpl::TensorImpl(DataType dtype, int32_t capacity)
    : type_(dtype),
      size_(0),
      int32_buf_(nullptr),
      int64_buf_(nullptr),
      float_buf_(nullptr),
      double_buf_(nullptr),
      string_buf_(nullptr) {
  if (type_ == DataType::kInt32) {
    int32_buf_ = new ::PB_NAMESPACE::RepeatedField<int32_t>();
    int32_buf_->Reserve(capacity);
  } else if (type_ == DataType::kInt64) {
    int64_buf_ = new ::PB_NAMESPACE::RepeatedField<int64_t>();
    int64_buf_->Reserve(capacity);
  } else if (type_ == DataType::kFloat) {
    float_buf_ = new ::PB_NAMESPACE::RepeatedField<float>();
    float_buf_->Reserve(capacity);
  } else if (type_ == DataType::kDouble) {
    double_buf_ = new ::PB_NAMESPACE::RepeatedField<double>();
    double_buf_->Reserve(capacity);
  } else if (type_ == DataType::kString) {
    string_buf_ = new ::PB_NAMESPACE::RepeatedPtrField<std::string>();
    string_buf_->Reserve(capacity);
  } else {
    LOG(ERROR) << "Invalid data type: " << static_cast<int32_t>(dtype);
  }
}

TensorImpl::~TensorImpl() {
  delete int32_buf_;
  delete int64_buf_;
  delete float_buf_;
  delete double_buf_;
  delete string_buf_;
}

TensorImpl::TensorImpl(TensorImpl&& other) noexcept
    : type_(other.type_), size_(other.size_),
      int32_buf_(other.int32_buf_),
      int64_buf_(other.int64_buf_),
      float_buf_(other.float_buf_),
      double_buf_(other.double_buf_),
      string_buf_(other.string_buf_) {
  other.type_ = DataType::kUnknown;
  other.size_ = 0;
  other.int32_buf_ = nullptr;
  other.int64_buf_ = nullptr;
  other.float_buf_ = nullptr;
  other.double_buf_ = nullptr;
  other.string_buf_ = nullptr;
}

TensorImpl& TensorImpl::operator=(TensorImpl&& other) noexcept {
  if (this != &other) {
    type_ = other.type_;
    size_ = other.size_;
    int32_buf_ = other.int32_buf_;
    int64_buf_ = other.int64_buf_;
    float_buf_ = other.float_buf_;
    double_buf_ = other.double_buf_;
    string_buf_ = other.string_buf_;
    other.type_ = DataType::kUnknown;
    other.size_ = 0;
    other.int32_buf_ = nullptr;
    other.int64_buf_ = nullptr;
    other.float_buf_ = nullptr;
    other.double_buf_ = nullptr;
    other.string_buf_ = nullptr;
  }
  return *this;
}


void TensorImpl::SwapWithProto(TensorValue* v) {
  if (type_ == DataType::kInt32) {
    auto tmp = v->mutable_int32_values();
    int32_buf_->Swap(tmp);
    size_ = int32_buf_->size();
  } else if (type_ == DataType::kInt64) {
    auto tmp = v->mutable_int64_values();
    int64_buf_->Swap(tmp);
    size_ = int64_buf_->size();
  } else if (type_ == DataType::kFloat) {
    auto tmp = v->mutable_float_values();
    float_buf_->Swap(tmp);
    size_ = float_buf_->size();
  } else if (type_ == DataType::kDouble) {
    auto tmp = v->mutable_double_values();
    double_buf_->Swap(tmp);
    size_ = double_buf_->size();
  } else if (type_ == DataType::kString) {
    auto tmp = v->mutable_string_values();
    string_buf_->Swap(tmp);
    size_ = string_buf_->size();
  } else {
    LOG(ERROR) << "Invalid data type: " << static_cast<int32_t>(type_);
  }
}

}  // namespace graphlearn
