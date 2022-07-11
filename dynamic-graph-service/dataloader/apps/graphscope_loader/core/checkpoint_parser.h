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

#ifndef GRAPHSCOPE_LOADER_CORE_CHECKPOINT_PARSER_H_
#define GRAPHSCOPE_LOADER_CORE_CHECKPOINT_PARSER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "dataloader/typedefs.h"

namespace dgs {
namespace dataloader {
namespace gs {

struct BulkLoadingInfo {
  std::string db_name;
  std::string schema_file;
  int64_t snapshot_id = 0;

  BulkLoadingInfo() = default;
  BulkLoadingInfo(const std::string& db_name, const std::string& schema_file, int64_t snapshot_id)
    : db_name(db_name), schema_file(schema_file), snapshot_id(snapshot_id) {}
  BulkLoadingInfo(const BulkLoadingInfo&) = default;
  BulkLoadingInfo& operator=(const BulkLoadingInfo&) = default;
  BulkLoadingInfo(BulkLoadingInfo&&) = default;
  BulkLoadingInfo& operator=(BulkLoadingInfo&&) = default;
};

std::vector<BulkLoadingInfo> GetBulkLoadingInfos(const std::string& restored_dir);
std::unordered_map<int32_t, DataStreamOffset> GetLogOffsets(const std::string& restored_dir);

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs

#endif // GRAPHSCOPE_LOADER_CORE_CHECKPOINT_PARSER_H_
