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

#include "graphlearn/core/graph/storage/storage_mode.h"
#include "graphlearn/include/config.h"

namespace graphlearn {
namespace io {

namespace {
  int32_t kCompressedMode = 1;
  int32_t kDataDistributionEnabled = 2;
  int32_t kVineyardEnabled = 8;
}  // anonymous namespace

bool IsCompressedStorageEnabled() {
  return GLOBAL_FLAG(StorageMode) & kCompressedMode;
}

bool IsDataDistributionEnabled() {
  return GLOBAL_FLAG(StorageMode) & kDataDistributionEnabled;
}

bool IsVineyardStorageEnabled() {
  return GLOBAL_FLAG(StorageMode) == kVineyardEnabled;
}

}  // namespace io
}  // namespace graphlearn
