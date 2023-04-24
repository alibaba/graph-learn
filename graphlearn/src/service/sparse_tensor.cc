/* Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "include/sparse_tensor.h"

namespace graphlearn {

SparseTensor::SparseTensor() {
}

SparseTensor::SparseTensor(const Tensor& segments, const Tensor& values)
  : segments_(segments),
    values_(values) {
}

SparseTensor::SparseTensor(Tensor&& segments, Tensor&& values)
  : segments_(segments),
    values_(values) {
}

SparseTensor::SparseTensor(const SparseTensor& other) noexcept
  : segments_(other.segments_),
    values_(other.values_) {
}

SparseTensor::SparseTensor(SparseTensor&& other) noexcept
    : segments_(std::move(other.segments_)),
      values_(std::move(other.values_)) {
}

SparseTensor& SparseTensor::operator=(const SparseTensor& other) noexcept {
  if (this != &other) {
    segments_ = other.segments_;
    values_ = other.values_;
  }
  return *this;
}

SparseTensor& SparseTensor::operator=(SparseTensor&& other) noexcept {
  if (this != &other) {
    segments_ = std::move(other.segments_);
    values_ = std::move(other.values_);
  }
  return *this;
}

SparseTensor::~SparseTensor() {
}

const Tensor& SparseTensor::Segments() const {
  return segments_;
}

const Tensor& SparseTensor::Values() const {
  return values_;
}

Tensor* SparseTensor::MutableSegments() {
  return &segments_;
}

Tensor* SparseTensor::MutableValues() {
  return &values_;
}

void SparseTensor::Swap(SparseTensor& right) {
  segments_.Swap(right.segments_);
  values_.Swap(right.values_);
}

void SparseTensor::SwapWithProto(SparseTensorValue* pb) {
  TensorValue* segments = pb->mutable_segments();
  segments->set_name("segments");
  segments->set_length(segments_.Size());
  segments->set_dtype(static_cast<int32_t>(segments_.DType()));
  segments_.SwapWithProto(segments);

  TensorValue* values = pb->mutable_values();
  values->set_name("values");
  values->set_length(values_.Size());
  values->set_dtype(static_cast<int32_t>(values_.DType()));
  values_.SwapWithProto(values);
}

}  // namespace graphlearn
