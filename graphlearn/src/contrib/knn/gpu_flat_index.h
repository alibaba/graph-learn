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

#ifndef GRAPHLEARN_CONTRIB_KNN_GPU_FLAT_INDEX_H_
#define GRAPHLEARN_CONTRIB_KNN_GPU_FLAT_INDEX_H_

#include <mutex>  //NOLINT [build/c++11]
#include <vector>
#include "contrib/knn/flat_index.h"
#include "contrib/knn/index.h"

namespace faiss {
namespace gpu {
  struct GpuIndexFlat;
  struct GpuResources;
}
}

namespace graphlearn {
namespace op {

#if defined(OPEN_GPU)
class GpuFlatKnnIndex : public KnnIndex {
public:
  explicit GpuFlatKnnIndex(int32_t d);
  virtual ~GpuFlatKnnIndex();

  void Train(size_t n, const float* data) override;

  void Add(size_t n, const float* data, const int64_t* ids) override;

  void Search(size_t n, const float* input, int32_t k,
              int64_t* ids, float* distances) const override;

protected:
  faiss::gpu::GpuIndexFlat* index_;
  faiss::gpu::GpuResources* res_;
  std::vector<int64_t>      ids_;
  mutable std::mutex        mtx_;
};
#else
class GpuFlatKnnIndex : public FlatKnnIndex {
public:
  explicit GpuFlatKnnIndex(int32_t d);
  virtual ~GpuFlatKnnIndex();
};
#endif

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CONTRIB_KNN_GPU_FLAT_INDEX_H_
