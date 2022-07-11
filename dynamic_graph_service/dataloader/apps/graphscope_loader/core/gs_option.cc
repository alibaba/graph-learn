/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "gs_option.h"

#include "dataloader/logging.h"
#include "dataloader/utils.h"

namespace dgs {
namespace dataloader {
namespace gs {

bool GSOptions::LoadFile(const std::string& option_file) {
  YAML::Node node;
  try {
    node = YAML::LoadFile(option_file);
  } catch (const YAML::ParserException& e) {
    LOG(ERROR) << "Loading malformed option file " << option_file << ": " << e.what();
    return false;
  } catch (const YAML::BadFile& e) {
    LOG(ERROR) << "Cannot load option file " << option_file << ": " << e.what();
    return false;
  }
  return SetOptions(node);
}

bool GSOptions::SetOptions(const YAML::Node& map) {
  bool suc = true;
  auto gs_map = map["graphscope"];
  if (gs_map) {
    auto polling_map = gs_map["log-polling"];
    if (polling_map) {
      std::string group = "graphscope-log-polling-option";
      suc &= SetListOption(source_kafka_brokers, polling_map, group, "kafka-brokers");
      suc &= SetScalarOption(source_kafka_topic, polling_map, group, "kafka-topic");
      suc &= SetScalarOption(source_kafka_partitions, polling_map, group, "kafka-partition-num");
      suc &= SetScalarOption(polling_meta_dir, polling_map, group, "meta-dir");
      suc &= SetScalarOption(polling_offset_persist_ms, polling_map, group, "offset-persist-ms");
      suc &= SetScalarOption(polling_retry_ms, polling_map, group, "retry-ms");
      suc &= SetScalarOption(polling_flush_ms, polling_map, group, "flush-ms");
    }
    auto bulk_loading_map = gs_map["bulk-loading"];
    if (bulk_loading_map) {
      std::string group = "graphscope-bulk-loading-option";
      suc &= SetScalarOption(bulk_loading_threads, bulk_loading_map, group, "thread-num");
      suc &= SetScalarOption(checkpoint_restore_dir, bulk_loading_map, group, "checkpoint-restore-dir");
      suc &= SetScalarOption(bulk_load_meta_dir, bulk_loading_map, group, "meta-dir");
    }
  }
  return suc;
}

std::string GSOptions::FormatSourceKafkaBrokers() const {
  return StrJoin(source_kafka_brokers, ",");
}

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs
