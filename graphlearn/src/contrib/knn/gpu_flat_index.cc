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

#include "contrib/knn/gpu_flat_index.h"

#include "contrib/knn/config.h"

#if defined(OPEN_GPU)
#include "faiss/gpu/GpuIndexFlat.h"
#include "faiss/gpu/StandardGpuResources.h"
#endif

namespace graphlearn {
namespace op {

#if defined(OPEN_GPU)

GpuFlatKnnIndex::GpuFlatKnnIndex(int32_t d) : KnnIndex(d) {
  res_ = new faiss::gpu::StandardGpuResources();
  faiss::gpu::GpuIndexFlatConfig config;
  config.device = 0;  // default the first GPU card
  if (IsL2Metric()) {
    index_ = new faiss::gpu::GpuIndexFlatIP(res_, d_, config);
  } else {
    index_ = new faiss::gpu::GpuIndexFlatL2(res_, d_, config);
  }
}

GpuFlatKnnIndex::~GpuFlatKnnIndex() {
  delete index_;
  delete res_;
}

void GpuFlatKnnIndex::Train(size_t n, const float* data) {
  index_->train(n, data);
}

void GpuFlatKnnIndex::Add(size_t n, const float* data, const int64_t* ids) {
  // Flat index treat index as ids. We should do the conversion.
  index_->add(n, data);
  ids_.reserve(n);
  ids_.insert(ids_.end(), ids, ids + n);
}

void GpuFlatKnnIndex::Search(size_t n, const float* input, int32_t k,
                             int64_t* ids, float* distances) const {
  size_t length = n * k;
  std::unique_ptr<int64_t[]> row_index(new int64_t[length]);
  {
    // GPU index is not thread-safe
    ScopedLocker<std::mutex> _(&mtx_);
    index_->search(n, input, k, distances, row_index.get());
  }
  for (size_t i = 0; i < length; ++i) {
    ids[i] = ids_[row_index[i]];
  }
}

#else

GpuFlatKnnIndex::GpuFlatKnnIndex(int32_t d) : FlatKnnIndex(d) {
}

GpuFlatKnnIndex::~GpuFlatKnnIndex() {
}

#endif

}  // namespace op
}  // namespace graphlearn
