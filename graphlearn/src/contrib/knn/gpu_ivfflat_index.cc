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

#include "contrib/knn/gpu_ivfflat_index.h"

#include "contrib/knn/config.h"

#if defined(OPEN_GPU)
#include "faiss/gpu/GpuIndexIVFFlat.h"
#include "faiss/gpu/StandardGpuResources.h"
#endif

namespace graphlearn {
namespace op {

#if defined(OPEN_GPU)

GpuIVFFlatKnnIndex::GpuIVFFlatKnnIndex(int32_t d, int32_t nlist, int32_t nprobe)
    : KnnIndex(d), nlist_(nlist), nprobe_(nprobe) {
  res_ = new faiss::gpu::StandardGpuResources();
  faiss::gpu::GpuIndexIVFFlatConfig config;
  config.device = 0;
  auto metric = IsL2Metric() ?
    faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
  index_ = new faiss::gpu::GpuIndexIVFFlat(res_, d_, nlist_, metric, config);
}

GpuIVFFlatKnnIndex::~GpuIVFFlatKnnIndex() {
  delete index_;
  delete res_;
}

void GpuIVFFlatKnnIndex::Train(size_t n, const float* data) {
  index_->train(n, data);
}

void GpuIVFFlatKnnIndex::Add(size_t n, const float* data, const int64_t* ids) {
  index_->add_with_ids(n, data, ids);
  index_->setNumProbes(nprobe_);
}

void GpuIVFFlatKnnIndex::Search(size_t n, const float* input, int32_t k,
                                int64_t* ids, float* distances) const {
  // GPU index is not thread-safe
  ScopedLocker<std::mutex> _(&mtx_);
  index_->search(n, input, k, distances, ids);
}

#else

GpuIVFFlatKnnIndex::GpuIVFFlatKnnIndex(int32_t d, int32_t nlist, int32_t nprobe)
  : IVFFlatKnnIndex(d, nlist, nprobe) {
}

GpuIVFFlatKnnIndex::~GpuIVFFlatKnnIndex() {
}

#endif

}  // namespace op
}  // namespace graphlearn
