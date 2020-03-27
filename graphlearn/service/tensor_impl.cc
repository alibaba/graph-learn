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

#include "graphlearn/service/tensor_impl.h"

#include "graphlearn/common/base/log.h"

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
    int32_buf_ = new ::google::protobuf::RepeatedField<int32_t>();
  } else if (type_ == DataType::kInt64) {
    int64_buf_ = new ::google::protobuf::RepeatedField<int64_t>();
  } else if (type_ == DataType::kFloat) {
    float_buf_ = new ::google::protobuf::RepeatedField<float>();
  } else if (type_ == DataType::kDouble) {
    double_buf_ = new ::google::protobuf::RepeatedField<double>();
  } else if (type_ == DataType::kString) {
    string_buf_ = new ::google::protobuf::RepeatedField<std::string>();
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
    int32_buf_ = new ::google::protobuf::RepeatedField<int32_t>();
    int32_buf_->Reserve(capacity);
  } else if (type_ == DataType::kInt64) {
    int64_buf_ = new ::google::protobuf::RepeatedField<int64_t>();
    int64_buf_->Reserve(capacity);
  } else if (type_ == DataType::kFloat) {
    float_buf_ = new ::google::protobuf::RepeatedField<float>();
    float_buf_->Reserve(capacity);
  } else if (type_ == DataType::kDouble) {
    double_buf_ = new ::google::protobuf::RepeatedField<double>();
    double_buf_->Reserve(capacity);
  } else if (type_ == DataType::kString) {
    string_buf_ = new ::google::protobuf::RepeatedField<std::string>();
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

void TensorImpl::SwapWithPB(void* pb) {
  if (type_ == DataType::kInt32) {
    ::google::protobuf::RepeatedField<int32_t>* pb_buf =
      static_cast<::google::protobuf::RepeatedField<int32_t>*>(pb);
    int32_buf_->Swap(pb_buf);
    size_ = int32_buf_->size();
  } else if (type_ == DataType::kInt64) {
    ::google::protobuf::RepeatedField<int64_t>* pb_buf =
      static_cast<::google::protobuf::RepeatedField<int64_t>*>(pb);
    int64_buf_->Swap(pb_buf);
    size_ = int64_buf_->size();
  } else if (type_ == DataType::kFloat) {
    ::google::protobuf::RepeatedField<float>* pb_buf =
      static_cast<::google::protobuf::RepeatedField<float>*>(pb);
    float_buf_->Swap(pb_buf);
    size_ = float_buf_->size();
  } else if (type_ == DataType::kDouble) {
    ::google::protobuf::RepeatedField<double>* pb_buf =
      static_cast<::google::protobuf::RepeatedField<double>*>(pb);
    double_buf_->Swap(pb_buf);
    size_ = double_buf_->size();
  } else if (type_ == DataType::kString) {
    ::google::protobuf::RepeatedField<std::string>* pb_buf =
      static_cast<::google::protobuf::RepeatedField<std::string>*>(pb);
    string_buf_->Swap(pb_buf);
    size_ = string_buf_->size();
  } else {
    LOG(ERROR) << "Invalid data type: " << static_cast<int32_t>(type_);
  }
}

void TensorImpl::CopyFromPB(const void* pb) {
  if (type_ == DataType::kInt32) {
    const ::google::protobuf::RepeatedField<int32_t>* pb_buf =
      static_cast<const ::google::protobuf::RepeatedField<int32_t>*>(pb);
    int32_buf_->CopyFrom(*pb_buf);
    size_ = int32_buf_->size();
  } else if (type_ == DataType::kInt64) {
    const ::google::protobuf::RepeatedField<int64_t>* pb_buf =
      static_cast<const ::google::protobuf::RepeatedField<int64_t>*>(pb);
    int64_buf_->CopyFrom(*pb_buf);
    size_ = int64_buf_->size();
  } else if (type_ == DataType::kFloat) {
    const ::google::protobuf::RepeatedField<float>* pb_buf =
      static_cast<const ::google::protobuf::RepeatedField<float>*>(pb);
    float_buf_->CopyFrom(*pb_buf);
    size_ = float_buf_->size();
  } else if (type_ == DataType::kDouble) {
    const ::google::protobuf::RepeatedField<double>* pb_buf =
      static_cast<const ::google::protobuf::RepeatedField<double>*>(pb);
    double_buf_->CopyFrom(*pb_buf);
    size_ = double_buf_->size();
  } else if (type_ == DataType::kString) {
    const ::google::protobuf::RepeatedField<std::string>* pb_buf =
      static_cast<const ::google::protobuf::RepeatedField<std::string>*>(pb);
    string_buf_->CopyFrom(*pb_buf);
    size_ = string_buf_->size();
  } else {
    LOG(ERROR) << "Invalid data type: " << static_cast<int32_t>(type_);
  }
}

}  // namespace graphlearn
