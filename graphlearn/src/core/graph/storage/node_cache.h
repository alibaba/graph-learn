/* Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_NODE_CACHE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_NODE_CACHE_H_

#include <mutex>
#include <thread>
#include "core/graph/storage/cache_policy.h"

namespace graphlearn {
namespace io {

template <typename Key, typename Value, template <typename ELEM> class Policy,
  typename key_type = typename std::enable_if<std::is_base_of<CachePolicy<Key>, Policy<Key>>::value>::type>
class NodeCache {
public:
  NodeCache(size_t max_cache_size, Policy<Key>&& policy)
  :max_cache_size_(max_cache_size), cache_policy_(std::move(policy)), current_cache_size_(0) {
    if(max_cache_size_ == 0) {
      throw std::invalid_argument("Max size of the cache should be greather than zero.");
      }
  }

  ~NodeCache() = default;

  void Insert(Key key, const Value& value) {
    std::lock_guard<std::mutex> lock(lock_);
    if (current_cache_size_ >= max_cache_size_) {
      const Key& last_item_key = cache_policy_.Eliminate();
      cache_items_.erase(last_item_key);
      cache_policy_.Erase(last_item_key);
      current_cache_size_--;
    }

    auto iter = cache_items_.find(key);
    if(iter != cache_items_.end()) {
      // update existing value
      iter->second = value;
      cache_policy_.Visit(key);
    } else {
      cache_items_.emplace(key, value);
      cache_policy_.Insert(key);
    }
    current_cache_size_++;
  }

  bool TryGet(const Key& key, Value& value) {
    std::lock_guard<std::mutex> lock(lock_);
    auto iter = cache_items_.find(key);
    if(iter != cache_items_.end()) {
      value = iter->second;
      cache_policy_.Visit(key);
      return true;
    }

    return false;
  }

  bool Contains(const Key& key) {
    std::lock_guard<std::mutex> lock(lock_);
    return cache_items_.find(key) != cache_items_.end();
  }

private:
  std::unordered_map<Key, Value> cache_items_;
  Policy<Key> cache_policy_;
  const size_t max_cache_size_;
  size_t current_cache_size_;
  std::mutex lock_;
};

}  // namespace io
}  // namespace graphlearn


#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_NODE_CACHE_H_