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

#include "checkpoint_parser.h"

#include <fstream>

#include "boost/filesystem.hpp"
#include "dataloader/logging.h"
#include "dataloader/utils.h"

namespace fs = boost::filesystem;

namespace dgs {
namespace dataloader {
namespace gs {

std::vector<BulkLoadingInfo> GetBulkLoadingInfos(const std::string& restored_dir) {
  fs::path restored_path(restored_dir);
  if (!fs::exists(restored_path)) {
    throw std::runtime_error("the maxgraph restored dir " + restored_dir + " is not existed.");
  }
  fs::path meta_path = restored_path / "meta";
  if (!fs::exists(meta_path)) {
    LOG(WARNING) << "Miss maxgraph restore meta files, skip bulk loading ..." ;
    return {};
  }
  fs::path store_path = restored_path / "store";
  if (!fs::exists(store_path)) {
    LOG(WARNING) << "Miss maxgraph restore store files, skip bulk loading ..." ;
    return {};
  }

  std::string schema_file = (meta_path / "graph_def_proto_bytes").string();
  std::string snapshot_file = (meta_path / "query_snapshot_id").string();
  int64_t snapshot_id;

  std::ifstream infile;
  infile.open(snapshot_file);
  infile >> snapshot_id;
  infile.close();

  std::vector<BulkLoadingInfo> infos;
  fs::recursive_directory_iterator iter(store_path);
  fs::recursive_directory_iterator end_iter;
  while (iter != end_iter) {
    if (fs::is_directory(*iter)) {
      infos.emplace_back((*iter).path().string(), schema_file, snapshot_id);
    }
    iter++;
  }
  return infos;
}

std::unordered_map<int32_t, DataStreamOffset> GetLogOffsets(const std::string& restored_dir) {
  fs::path restored_path(restored_dir);
  fs::path meta_path = restored_path / "meta";
  if (!fs::exists(meta_path / "queue_offsets")) {
    LOG(WARNING) << "Missing maxgraph queue offsets file, will poll from offset 0.";
    return {};
  }

  std::string offsets_file = (meta_path / "queue_offsets").string();
  std::string offsets_str;
  std::ifstream infile;
  infile.open(offsets_file);
  infile >> offsets_str;
  infile.close();

  std::unordered_map<int32_t, DataStreamOffset> log_offsets;
  auto offsets = StrSplit(offsets_str.substr(1, offsets_str.size() - 2), ',');
  for (int32_t i = 0; i < offsets.size(); i++) {
    log_offsets.emplace(i, std::stoll(offsets[i]));
  }
  return log_offsets;
}

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs