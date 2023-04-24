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

#include "core/dag/tensor_map.h"

#include <memory>
#include <unordered_map>

namespace graphlearn {

TensorMap::TensorMap() {}

TensorMap::TensorMap(Tensor::Map&& tensor,
                     SparseTensor::Map&& sparse_tensor)
  : tensors_(std::move(tensor)),
    sparse_tensors_(std::move(sparse_tensor)) {
}

TensorMap::TensorMap(TensorMap&& other) noexcept
  : tensors_(std::move(other.tensors_)),
    sparse_tensors_(std::move(other.sparse_tensors_)) {
}

TensorMap::~TensorMap() {
  tensors_.clear();
  sparse_tensors_.clear();
}

TensorMap& TensorMap::operator=(TensorMap&& other) noexcept {
  if (this != &other) {
    tensors_ = std::move(other.tensors_);
    sparse_tensors_ = std::move(other.sparse_tensors_);
  }
  return *this;
}

const int32_t TensorMap::Size() const {
  return tensors_.size() + sparse_tensors_.size();
}

std::pair<const Tensor*, const Tensor*> TensorMap::Find(
    const std::string& key) {
  auto iter = tensors_.find(key);
  if (iter == tensors_.end()) {
    auto iter_sp = sparse_tensors_.find(key);
    if (iter_sp == sparse_tensors_.end()) {
      return {nullptr, nullptr};
    }
    const Tensor* values = &iter_sp->second.Values();
    const Tensor* segments = &iter_sp->second.Segments();
    return {values, segments};
  }
  const Tensor* values = &iter->second;
  return {values, nullptr};
}

bool TensorMap::Add(const std::string& key,
                    const Tensor* values,
                    const Tensor* segments) {
  if (!values && !segments) {
    return false;
  }
  if (segments) {
    sparse_tensors_.emplace(key, SparseTensor(*segments, *values));
  } else {
    tensors_.emplace(key, Tensor(*values));
  }
  return true;
}

bool TensorMap::Add(const std::string& key,
                    Tensor&& values,
                    Tensor&& segments) {
  if (values.Size() == 0) {
    return false;
  }
  if (segments.Size() > 0) {
    sparse_tensors_.emplace(
      key, SparseTensor{std::move(segments), std::move(values)});
  } else {
    tensors_.emplace(key, Tensor{std::move(values)});
  }
  return true;
}

}  // namespace graphlearn