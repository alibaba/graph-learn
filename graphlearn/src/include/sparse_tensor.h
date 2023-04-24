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

#ifndef GRAPHLEARN_INCLUDE_SPARSE_TENSOR_H_
#define GRAPHLEARN_INCLUDE_SPARSE_TENSOR_H_

#include "include/tensor.h"

namespace graphlearn {

/// `SparseTensor` is used exclusively in GraphLearn to express
/// the sparsity of graph. `segments` is how to split the `values`.
/// Usually it represents the neighbors of a batch of nodes,
/// where `values` is the continuous storage of neighbors of each node
/// within the batch, and `segments` is the number of neighbors of each node.
class SparseTensor {
public:
  SparseTensor();
  SparseTensor(const Tensor& segments, const Tensor& vals);
  SparseTensor(Tensor&& segments, Tensor&& vals);

  SparseTensor(const SparseTensor& other) noexcept;
  SparseTensor(SparseTensor&& other) noexcept;
  SparseTensor& operator=(const SparseTensor& other) noexcept;
  SparseTensor& operator=(SparseTensor&& other) noexcept;
  ~SparseTensor();

  const Tensor& Segments() const;
  const Tensor& Values() const;

  Tensor* MutableSegments();
  Tensor* MutableValues();

  void Swap(SparseTensor& right);
  void SwapWithProto(SparseTensorValue* pb);

  typedef std::unordered_map<std::string, SparseTensor> Map;

protected:
  Tensor segments_;
  Tensor values_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_SPARSE_TENSOR_H_
