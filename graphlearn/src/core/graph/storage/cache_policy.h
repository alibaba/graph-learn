/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_CACHE_POLICY_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_CACHE_PLOICY_H_

#include <map>
#include <unordered_map>
#include <iostream>

namespace graphlearn {
namespace io {

template <typename Key>
struct CachePolicy {
public:
  virtual ~CachePolicy() = default;
  virtual void Insert(const Key &key) = 0;
  virtual void Visit(const Key &key) = 0;
  virtual void Erase(const Key &key) = 0;
  virtual const Key& Eliminate() = 0 ;
};

template <typename Key>
class LFUCachePolicy :public CachePolicy<Key> {
public:
  using lfu_iterator = typename std::multimap<std::size_t, Key>::iterator;
  LFUCachePolicy() = default;
  LFUCachePolicy(const LFUCachePolicy&) = default;
  LFUCachePolicy(LFUCachePolicy&&) = default;

  virtual void Insert(const Key& key) {
    if (item_frequency_map_.find(key) != item_frequency_map_.end()) {
      Visit(key);
      return;
    }

    constexpr std::size_t INIT_VAL = 1;
    item_frequency_map_[key] = frequency_item_map_.emplace(INIT_VAL, key);
  }

  virtual void Visit(const Key& key) {
    auto ite = item_frequency_map_.find(key);
    if(ite != item_frequency_map_.end()) {
          auto fre = ite->second->first + 1;
          frequency_item_map_.erase(ite->second);
          item_frequency_map_[key] = frequency_item_map_.emplace_hint(frequency_item_map_.cend(), fre, key);
    }
  }

  virtual void Erase(const Key& key) {
    auto ite = item_frequency_map_.find(key);
    if(ite != item_frequency_map_.end()) {
      frequency_item_map_.erase(ite->second);
      item_frequency_map_.erase(ite);
    }
  }

  virtual const Key& Eliminate() {
    return frequency_item_map_.cbegin()->second;
  }

private:
  std::unordered_map<Key, lfu_iterator> item_frequency_map_;
  std::multimap<std::size_t, Key> frequency_item_map_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_CACHE_PLOICY_H_