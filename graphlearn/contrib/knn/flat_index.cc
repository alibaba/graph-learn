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

#include "graphlearn/contrib/knn/flat_index.h"

#include <memory>
#include "graphlearn/contrib/knn/config.h"

#include "faiss/IndexFlat.h"

namespace graphlearn {
namespace op {

FlatKnnIndex::FlatKnnIndex(int32_t d) : KnnIndex(d) {
  auto metric = IsL2Metric() ?
    faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
  index_ = new faiss::IndexFlat(d_, metric);
}

FlatKnnIndex::~FlatKnnIndex() {
  delete index_;
}

void FlatKnnIndex::Train(size_t n, const float* data) {
  // Flat index does not need train
}

void FlatKnnIndex::Add(size_t n, const float* data, const int64_t* ids) {
  // Flat index treat index as ids. We should do the conversion.
  index_->add(n, data);
  ids_.reserve(n);
  ids_.insert(ids_.end(), ids, ids + n);
}

void FlatKnnIndex::Search(size_t n, const float* input, int32_t k,
                          int64_t* ids, float* distances) const {
  size_t length = n * k;
  std::unique_ptr<int64_t[]> row_index(new int64_t[length]);
  index_->search(n, input, k, distances, row_index.get());
  for (size_t i = 0; i < length; ++i) {
    ids[i] = ids_[row_index[i]];
  }
}

}  // namespace op
}  // namespace graphlearn
