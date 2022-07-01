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

#include "graphlearn/contrib/knn/index_factory.h"

#include "graphlearn/common/base/log.h"
#include "graphlearn/contrib/knn/flat_index.h"
#include "graphlearn/contrib/knn/gpu_flat_index.h"
#include "graphlearn/contrib/knn/gpu_ivfflat_index.h"
#include "graphlearn/contrib/knn/gpu_ivfpq_index.h"
#include "graphlearn/contrib/knn/ivfflat_index.h"
#include "graphlearn/contrib/knn/ivfpq_index.h"

namespace graphlearn {
namespace op {

KnnIndex* KnnIndexFactory::Create(const IndexOption& option) {
  if (option.index_type == "flat") {
    return new FlatKnnIndex(option.dimension);
  } else if (option.index_type == "ivfflat") {
    return new IVFFlatKnnIndex(
      option.dimension, option.nlist, option.nprobe);
  } else if (option.index_type == "ivfpq") {
    return new IVFPQKnnIndex(
      option.dimension, option.nlist, option.nprobe, option.m);
  } else if (option.index_type == "gpu_flat") {
    return new GpuFlatKnnIndex(option.dimension);
  } else if (option.index_type == "gpu_ivfflat") {
    return new GpuIVFFlatKnnIndex(
      option.dimension, option.nlist, option.nprobe);
  } else if (option.index_type == "gpu_ivfpq") {
    return new GpuIVFPQKnnIndex(
      option.dimension, option.nlist, option.nprobe, option.m);
  }
  USER_LOG("Invalid KNN index type: " + option.index_type);
  USER_LOG("flat/ivfflat/ivfpq/gpu_flat/gpu_ivfflat/gpu_ivfpq are supported.");
  return nullptr;
}

}  // namespace op
}  // namespace graphlearn
