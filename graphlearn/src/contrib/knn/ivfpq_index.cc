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

#include "contrib/knn/ivfpq_index.h"
#include "contrib/knn/config.h"

#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"

namespace graphlearn {
namespace op {

IVFPQKnnIndex::IVFPQKnnIndex(int32_t d, int32_t nlist,
                             int32_t nprobe, int32_t m)
  : KnnIndex(d), nlist_(nlist), nprobe_(nprobe), m_(m) {
  if (IsL2Metric()) {
    quantizer_ = new faiss::IndexFlatIP(d_);
    index_ = new faiss::IndexIVFPQ(quantizer_, d_, nlist_,
        m_, 8, faiss::METRIC_INNER_PRODUCT);
  } else {
    quantizer_ = new faiss::IndexFlatL2(d_);
    index_ = new faiss::IndexIVFPQ(quantizer_, d_, nlist_,
        m_, 8, faiss::METRIC_L2);
  }
}

IVFPQKnnIndex::~IVFPQKnnIndex() {
  delete index_;
  delete quantizer_;
}

void IVFPQKnnIndex::Train(size_t n, const float* data) {
  index_->train(n, data);
}

void IVFPQKnnIndex::Add(size_t n, const float* data, const int64_t* ids) {
  index_->add_with_ids(n, data, ids);
  index_->nprobe = nprobe_;
}

void IVFPQKnnIndex::Search(size_t n, const float* input, int32_t k,
                           int64_t* ids, float* distances) const {
  index_->search(n, input, k, distances, ids);
}

}  // namespace op
}  // namespace graphlearn
