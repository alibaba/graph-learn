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

#ifndef GRAPHLEARN_SERVICE_TENSOR_IMPL_H_
#define GRAPHLEARN_SERVICE_TENSOR_IMPL_H_

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "graphlearn/include/constants.h"
#include "google/protobuf/repeated_field.h"

namespace graphlearn {

class TensorImpl {
public:
  explicit TensorImpl(DataType dtype);
  TensorImpl(DataType dtype, int32_t capacity);
  ~TensorImpl();

  DataType DType() const {
    return type_;
  }

  int32_t Size() const {
    return size_;
  }

  void Resize(int32_t size) {
    if (type_ == DataType::kInt32) {
      int32_buf_->Resize(size, 0);
    } else if (type_ == DataType::kInt64) {
      int64_buf_->Resize(size, 0);
    } else if (type_ == DataType::kFloat) {
      float_buf_->Resize(size, 0.0);
    } else if (type_ == DataType::kDouble) {
      double_buf_->Resize(size, 0.0);
    } else if (type_ == DataType::kString) {
      string_buf_->Resize(size, "");
    } else {
    }
    size_ = size;
  }

  void AddInt32(int32_t v) {
    int32_buf_->Add(v);
    size_ = int32_buf_->size();
  }

  void AddInt64(int64_t v) {
    int64_buf_->Add(v);
    size_ = int64_buf_->size();
  }

  void AddFloat(float v) {
    float_buf_->Add(v);
    size_ = float_buf_->size();
  }

  void AddDouble(double v) {
    double_buf_->Add(v);
    size_ = double_buf_->size();
  }

  void AddString(const std::string& v) {
    string_buf_->Add(v);
    size_ = string_buf_->size();
  }

  void AddInt32(const int32_t* begin, const int32_t* end) {
    // int32_buf_->Add(begin, end);
    for (auto iter = begin; iter != end; ++iter) {
      int32_buf_->Add(*iter);
    }
    size_ = int32_buf_->size();
  }

  void AddInt64(const int64_t* begin, const int64_t* end) {
    // int64_buf_->Add(begin, end);
    for (auto iter = begin; iter != end; ++iter) {
      int64_buf_->Add(*iter);
    }
    size_ = int64_buf_->size();
  }

  void AddFloat(const float* begin, const float* end) {
    // float_buf_->Add(begin, end);
    for (auto iter = begin; iter != end; ++iter) {
      float_buf_->Add(*iter);
    }
    size_ = float_buf_->size();
  }

  void AddDouble(const double* begin, const double* end) {
    // double_buf_->Add(begin, end);
    for (auto iter = begin; iter != end; ++iter) {
      double_buf_->Add(*iter);
    }
    size_ = double_buf_->size();
  }

  void SetInt32(int32_t index, int32_t v) {
    int32_buf_->Set(index, v);
  }

  void SetInt64(int32_t index, int64_t v) {
    int64_buf_->Set(index, v);
  }

  void SetFloat(int32_t index, float v) {
    float_buf_->Set(index, v);
  }

  void SetDouble(int32_t index, double v) {
    double_buf_->Set(index, v);
  }

  void SetString(int32_t index, const std::string& v) {
    string_buf_->Set(index, v);
  }

  int32_t GetInt32(int32_t index) const {
    return int32_buf_->Get(index);
  }

  int64_t GetInt64(int32_t index) const {
    return int64_buf_->Get(index);
  }

  float   GetFloat(int32_t index) const {
    return float_buf_->Get(index);
  }

  double  GetDouble(int32_t index) const {
    return double_buf_->Get(index);
  }

  const std::string& GetString(int32_t index) const {
    return string_buf_->Get(index);
  }

  const int32_t* GetInt32() const {
    return int32_buf_->data();
  }

  const int64_t* GetInt64() const {
    return int64_buf_->data();
  }

  const float* GetFloat() const {
    return float_buf_->data();
  }

  const double* GetDouble() const {
    return double_buf_->data();
  }

  const std::string* GetString() const {
    return string_buf_->data();
  }

  void SwapWithPB(void* pb);
  void CopyFromPB(const void* pb);

private:
  TensorImpl(const TensorImpl& t);
  TensorImpl& operator=(const TensorImpl& t);

private:
  DataType type_;
  int32_t  size_;
  ::google::protobuf::RepeatedField<int32_t>*     int32_buf_;
  ::google::protobuf::RepeatedField<int64_t>*     int64_buf_;
  ::google::protobuf::RepeatedField<float>*       float_buf_;
  ::google::protobuf::RepeatedField<double>*      double_buf_;
  ::google::protobuf::RepeatedField<std::string>* string_buf_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_TENSOR_IMPL_H_
