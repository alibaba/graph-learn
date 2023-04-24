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

#ifndef GRAPHLEARN_ACTOR_TENSOR_SERIALIZER_H_
#define GRAPHLEARN_ACTOR_TENSOR_SERIALIZER_H_

#include "hiactor/net/serializable_queue.hh"

#include "core/dag/tape.h"
#include "include/tensor.h"

namespace graphlearn {
namespace act {

class TensorSerializer : public Tensor {
public:
  explicit TensorSerializer(const Tensor& t) : Tensor(t) {}
  ~TensorSerializer() = default;

  void dump_to(hiactor::serializable_queue& qu);
  static Tensor load_from(hiactor::serializable_queue& qu);
};

class TensorMapSerializer : public TensorMap {
public:
  TensorMapSerializer() : TensorMap() {}
  TensorMapSerializer(Tensor::Map&& tensor, SparseTensor::Map&& sparse_tensor)
    : TensorMap(std::move(tensor), std::move(sparse_tensor)) {}
  void dump_to(hiactor::serializable_queue &qu);  // NOLINT [runtime/references]
  static TensorMapSerializer load_from(hiactor::serializable_queue &qu);  // NOLINT [runtime/references]
};

struct TapeHolder {
  Tape* tape;

  explicit TapeHolder(Tape* t) : tape(t) {}
  ~TapeHolder() = default;

  // TapeHolder doesn't need (de-)serialization.
  void dump_to(hiactor::serializable_queue &qu) {}  // NOLINT [runtime/references]

  static TapeHolder load_from(hiactor::serializable_queue &qu) {  // NOLINT [runtime/references]
    return TapeHolder{nullptr};
  }
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_TENSOR_SERIALIZER_H_