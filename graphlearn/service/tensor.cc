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

#include "graphlearn/include/tensor.h"
#include "graphlearn/service/tensor_impl.h"

namespace graphlearn {

Tensor::Tensor() {
}

Tensor::Tensor(DataType dtype) {
  impl_.reset(new TensorImpl(dtype));
}

Tensor::Tensor(DataType dtype, int32_t capacity) {
  impl_.reset(new TensorImpl(dtype, capacity));
}

Tensor::Tensor(const Tensor& t) {
  impl_ = t.impl_;
}

Tensor& Tensor::operator=(const Tensor& t) {
  impl_ = t.impl_;
  return *this;
}

Tensor::~Tensor() {
}

DataType Tensor::DType() const {
  return impl_->DType();
}

int32_t Tensor::Size() const {
  return impl_->Size();
}

void Tensor::Resize(int32_t size) {
  return impl_->Resize(size);
}

void Tensor::AddInt32(int32_t v) {
  impl_->AddInt32(v);
}

void Tensor::AddInt64(int64_t v) {
  impl_->AddInt64(v);
}

void Tensor::AddFloat(float v) {
  impl_->AddFloat(v);
}

void Tensor::AddDouble(double v) {
  impl_->AddDouble(v);
}

void Tensor::AddString(const std::string& v) {
  impl_->AddString(v);
}

void Tensor::AddInt32(const int32_t* begin, const int32_t* end) {
  impl_->AddInt32(begin, end);
}

void Tensor::AddInt64(const int64_t* begin, const int64_t* end) {
  impl_->AddInt64(begin, end);
}

void Tensor::AddFloat(const float* begin, const float* end) {
  impl_->AddFloat(begin, end);
}

void Tensor::AddDouble(const double* begin, const double* end) {
  impl_->AddDouble(begin, end);
}

void Tensor::SetInt32(int32_t index, int32_t v) {
  impl_->SetInt32(index, v);
}

void Tensor::SetInt64(int32_t index, int64_t v) {
  impl_->SetInt64(index, v);
}

void Tensor::SetFloat(int32_t index, float v) {
  impl_->SetFloat(index, v);
}

void Tensor::SetDouble(int32_t index, double v) {
  impl_->SetDouble(index, v);
}

void Tensor::SetString(int32_t index, const std::string& v) {
  impl_->SetString(index, v);
}

int32_t Tensor::GetInt32(int32_t index) const {
  return impl_->GetInt32(index);
}

int64_t Tensor::GetInt64(int32_t index) const {
  return impl_->GetInt64(index);
}

float Tensor::GetFloat(int32_t index) const {
  return impl_->GetFloat(index);
}

double Tensor::GetDouble(int32_t index) const {
  return impl_->GetDouble(index);
}

const std::string& Tensor::GetString(int32_t index) const {
  return impl_->GetString(index);
}

const int32_t* Tensor::GetInt32() const {
  return impl_->GetInt32();
}

const int64_t* Tensor::GetInt64() const {
  return impl_->GetInt64();
}

const float* Tensor::GetFloat() const {
  return impl_->GetFloat();
}

const double* Tensor::GetDouble() const {
  return impl_->GetDouble();
}

const std::string* Tensor::GetString() const {
  return impl_->GetString();
}

void Tensor::Swap(Tensor& right) {
  std::shared_ptr<TensorImpl> tmp = right.impl_;
  right.impl_ = impl_;
  impl_ = tmp;
}

void Tensor::SwapWithPB(void* pb) {
  impl_->SwapWithPB(pb);
}

void Tensor::CopyFromPB(const void* pb) {
  impl_->CopyFromPB(pb);
}

}  // namespace graphlearn
