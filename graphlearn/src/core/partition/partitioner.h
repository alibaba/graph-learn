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

#ifndef GRAPHLEARN_CORE_PARTITION_PARTITIONER_H_
#define GRAPHLEARN_CORE_PARTITION_PARTITIONER_H_

#include <cstdint>
#include <memory>
#include <vector>
#include "core/partition/base_partitioner.h"
#include "core/partition/hash_partitioner.h"
#include "core/partition/no_partitioner.h"
#include "core/partition/stitcher.h"
#include "include/config.h"
#include "include/constants.h"
#include "platform/env.h"

namespace graphlearn {

template<class T>
class PartitionerCreator {
public:
  PartitionerCreator(int32_t range) {
    // If more partition strategies exist, register it here.
    no_parter_.reset(new NoPartitioner<T>());
    hash_parter_.reset(new HashPartitioner<T>(range));
  }
  ~PartitionerCreator() = default;

  BasePartitioner<T>* operator() (int32_t mode) {
    if (mode == kNoPartition) {
      return no_parter_.get();
    } else if (mode == kByHash) {
      return hash_parter_.get();
    }
    return no_parter_.get();
  }

private:
  std::unique_ptr<NoPartitioner<T>> no_parter_;
  std::unique_ptr<HashPartitioner<T>> hash_parter_;
};

template<class T>
BasePartitioner<T>* GetPartitioner(const T* t) {
  static int32_t n = Env::Default()->GetServerCount();
  static PartitionerCreator<T> creator(n);
  return creator(GLOBAL_FLAG(PartitionMode));
}

template<class T>
Stitcher<T>* GetStitcher(const T* t) {
  static Stitcher<T> s;
  return &s;
}

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_PARTITION_PARTITIONER_H_
