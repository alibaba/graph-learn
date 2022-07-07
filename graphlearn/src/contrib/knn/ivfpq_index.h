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

#ifndef GRAPHLEARN_CONTRIB_KNN_IVFPQ_INDEX_H_
#define GRAPHLEARN_CONTRIB_KNN_IVFPQ_INDEX_H_

#include <vector>
#include "contrib/knn/index.h"

namespace faiss {
  struct IndexFlat;
  struct IndexFlatIP;
  struct IndexFlatL2;
  struct IndexIVFPQ;
}

namespace graphlearn {
namespace op {

class IVFPQKnnIndex : public KnnIndex {
public:
  IVFPQKnnIndex(int32_t d, int32_t nlist, int32_t nprobe, int32_t m);
  virtual ~IVFPQKnnIndex();

  void Train(size_t n, const float* data) override;

  void Add(size_t n, const float* data, const int64_t* ids) override;

  void Search(size_t n, const float* input, int32_t k,
              int64_t* ids, float* distances) const override;

protected:
  faiss::IndexFlat*   quantizer_;
  faiss::IndexIVFPQ*  index_;
  int32_t             nlist_;
  int32_t             nprobe_;
  int32_t             m_;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CONTRIB_KNN_IVFPQ_INDEX_H_
