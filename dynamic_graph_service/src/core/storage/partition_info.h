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

#ifndef DGS_CORE_STORAGE_PARTITION_INFO_H_
#define DGS_CORE_STORAGE_PARTITION_INFO_H_

#include "common/actor_wrapper.h"
#include "common/typedefs.h"

namespace dgs {
namespace storage {

struct PartitionInfo {
  PartitionInfo() = default;
  ~PartitionInfo() = default;

  void dump_to(act::SerializableQueue &qu) {}  // NOLINT

  static PartitionInfo load_from(act::SerializableQueue &qu) {  // NOLINT
    return PartitionInfo();
  }

  std::vector<PartitionId> pids;
};

}  // namespace storage
}  // namespace dgs

#endif  // DGS_CORE_STORAGE_PARTITION_INFO_H_
