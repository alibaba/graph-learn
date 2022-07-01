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

#include "graphlearn/contrib/knn/index_manager.h"

#include "graphlearn/common/threading/sync/lock.h"

namespace graphlearn {
namespace op {

KnnIndexManager::KnnIndexManager() {
}

KnnIndexManager::~KnnIndexManager() {
  for (auto& v : m_) {
    delete v.second;
    v.second = nullptr;
  }
}

KnnIndexManager* KnnIndexManager::Instance() {
  static KnnIndexManager manager;
  return &manager;
}

void KnnIndexManager::Add(const std::string& data_type, KnnIndex* index) {
  ScopedLocker<std::mutex> _(&mtx_);
  auto it = m_.find(data_type);
  if (it != m_.end()) {
    return;
  }

  m_.insert(it, {data_type, index});
}

KnnIndex* KnnIndexManager::Get(const std::string& data_type) {
  ScopedLocker<std::mutex> _(&mtx_);
  auto it = m_.find(data_type);
  if (it == m_.end()) {
    return nullptr;
  }
  return it->second;
}

void KnnIndexManager::Remove(const std::string& data_type) {
  ScopedLocker<std::mutex> _(&mtx_);
  auto it = m_.find(data_type);
  if (it == m_.end()) {
    return;
  }
  delete it->second;
  m_.erase(it);
}

}  // namespace op
}  // namespace graphlearn
