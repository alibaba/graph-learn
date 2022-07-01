/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef DGS_CORE_EXECUTION_TENSOR_H_
#define DGS_CORE_EXECUTION_TENSOR_H_

#include "common/typedefs.h"
#include "core/io/record.h"

namespace dgs {
namespace execution {

struct Tensor {
  // Tensor is a shareble batch of ids for
  // input and output stream between DagNodes.
  actor::IdColumnBatch ids;

  explicit Tensor(size_t size) : ids(size) {}

  explicit Tensor(actor::IdColumnBatch&& input)
    : ids(std::move(input)) {
  }

  // TODO(wenting.swt): optimize with memcpy
  explicit Tensor(std::vector<VertexId>&& input)
    : ids(input.size()) {
    for (auto vid : input) {
      ids.push_back(vid);
    }
  }

  Tensor(const std::vector<io::Record>& records, FieldIndex idx)
    : ids(records.size()) {
    for (auto& record : records) {
      auto viewer = record.GetView();
      if (viewer.Type() == RecordType::VERTEX) {
        ids.push_back(viewer.AsVertexRecord().Id());
      } else {
        if (idx == 0) {
          ids.push_back(viewer.AsEdgeRecord().SrcId());
        } else {
          ids.push_back(viewer.AsEdgeRecord().DstId());
        }
      }
    }
  }

  Tensor(const Tensor& other) = delete;
  Tensor(Tensor&& other) : ids(std::move(other.ids)) {}

  void operator=(const Tensor& other) = delete;

  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      ids = std::move(other.ids);
    }
    return *this;
  }

  size_t Size() const {
    return ids.size();
  }

  Tensor Share() {
    return Tensor{ids.share()};
  }
};

}  // namespace execution
}  // namespace dgs

#endif  // DGS_CORE_EXECUTION_TENSOR_H_
