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

#ifndef GRAPHLEARN_INCLUDE_TENSOR_H_
#define GRAPHLEARN_INCLUDE_TENSOR_H_

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "graphlearn/include/constants.h"

namespace graphlearn {

class TensorImpl;

class Tensor {
public:
  Tensor();
  explicit Tensor(DataType dtype);
  Tensor(DataType dtype, int32_t capacity);
  Tensor(const Tensor& t);
  Tensor& operator=(const Tensor& t);
  ~Tensor();

  DataType DType() const;
  int32_t Size() const;
  void Resize(int32_t size);

  void AddInt32(int32_t v);
  void AddInt64(int64_t v);
  void AddFloat(float v);
  void AddDouble(double v);
  void AddString(const std::string& v);

  void AddInt32(const int32_t* begin, const int32_t* end);
  void AddInt64(const int64_t* begin, const int64_t* end);
  void AddFloat(const float* begin, const float* end);
  void AddDouble(const double* begin, const double* end);

  void SetInt32(int32_t index, int32_t v);
  void SetInt64(int32_t index, int64_t v);
  void SetFloat(int32_t index, float v);
  void SetDouble(int32_t index, double v);
  void SetString(int32_t index, const std::string& v);

  int32_t GetInt32(int32_t index) const;
  int64_t GetInt64(int32_t index) const;
  float   GetFloat(int32_t index) const;
  double  GetDouble(int32_t index) const;
  const std::string& GetString(int32_t index) const;

  const int32_t* GetInt32() const;
  const int64_t* GetInt64() const;
  const float*   GetFloat() const;
  const double*  GetDouble() const;
  const std::string* GetString() const;

  void Swap(Tensor& right);
  void SwapWithPB(void* pb);
  void CopyFromPB(const void* pb);

private:
  std::shared_ptr<TensorImpl> impl_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_TENSOR_H_
