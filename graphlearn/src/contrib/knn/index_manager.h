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

#ifndef GRAPHLEARN_CONTRIB_KNN_INDEX_MANAGER_H_
#define GRAPHLEARN_CONTRIB_KNN_INDEX_MANAGER_H_

#include <cstdint>
#include <mutex>  //NOLINT [build/c++11]
#include <string>
#include <unordered_map>

#include "contrib/knn/index.h"

namespace graphlearn {
namespace op {

class KnnIndexManager {
public:
  ~KnnIndexManager();
  static KnnIndexManager* Instance();

  /// Add an index, after which the manager will take the ownership of index.
  /// If the given type exists, do nothing.
  void Add(const std::string& data_type, KnnIndex* index);

  /// Get an index. If not exist, return nullptr.
  KnnIndex* Get(const std::string& data_type);

  /// Remove an index and release the memory. If not exist, do nothing.
  void Remove(const std::string& data_type);

private:
  KnnIndexManager();

private:
  std::mutex                                 mtx_;
  std::unordered_map<std::string, KnnIndex*> m_;
};

}  // namespace op
}  // namespace graphlearn

#endif  // GRAPHLEARN_CONTRIB_KNN_INDEX_MANAGER_H_
