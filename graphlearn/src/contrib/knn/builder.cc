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

#include "contrib/knn/builder.h"

#include <vector>
#include "contrib/knn/index_factory.h"
#include "contrib/knn/index_manager.h"

namespace graphlearn {

bool BuildKnnIndex(io::NodeStorage* storage, const IndexOption& option) {
  int32_t f_num = storage->GetSideInfo()->f_num;
  if (f_num < 1) {
    return false;
  }

  IndexOption copy = option;
  copy.dimension = f_num;
  op::KnnIndex* index = op::KnnIndexFactory::Create(copy);
  if (index == nullptr) {
    return false;
  }

  size_t n = storage->Size();
  std::vector<float> vectors(n * f_num);
  float* raw_vectors = const_cast<float*>(vectors.data());

  const std::vector<io::Attribute>* attrs = storage->GetAttributes();
  for (size_t i = 0; i < attrs->size(); ++i) {
    std::memcpy(raw_vectors, (*attrs)[i]->GetFloats(nullptr),
                f_num * sizeof(float));
    raw_vectors += f_num;
  }

  index->Train(n, vectors.data());
  index->Add(n, vectors.data(), storage->GetIds()->data());

  op::KnnIndexManager::Instance()->Add(storage->GetSideInfo()->type, index);
  return true;
}

}  // namespace graphlearn
