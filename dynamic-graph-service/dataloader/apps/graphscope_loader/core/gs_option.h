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

#ifndef GRAPHSCOPE_LOADER_GS_OPTION_H_
#define GRAPHSCOPE_LOADER_GS_OPTION_H_

#include "dataloader/options.h"

namespace dgs {
namespace dataloader {
namespace gs {

struct GSOptions {
  static GSOptions& GetInstance();

  /// Load gs options from yaml file.
  /// \return @True if successful.
  bool LoadFile(const std::string& option_file);

  /// The brokers of source kafka queues with streaming graph updates.
  std::vector<std::string> source_kafka_brokers = { "localhost:9092" };
  std::string FormatSourceKafkaBrokers() const;

  /// The name of source kafka topic.
  std::string source_kafka_topic = "graph_store";

  /// The partition number of source kafka topic.
  int32_t source_kafka_partitions = 2;

  /// The root dir to store log polling progress meta files.
  /// Meta files in root dir are distinguished by their corresponding source kafka partition id.
  std::string polling_meta_dir = "./polling_offsets";

  /// The interval(ms) for persisting current log polling progress.
  uint32_t polling_offset_persist_ms = 1000;

  /// The interval(ms) for retries after an invalid polling.
  uint32_t polling_retry_ms = 100;

  /// The max latency(ms) to flush current polled data into output queues.
  uint32_t polling_flush_ms = 100;

  /// The number of threads for bulk loading.
  uint32_t bulk_loading_threads = 2;

  /// The meta dir of bulk loading
  std::string bulk_load_meta_dir = "./bulk_load_flags";

  /// The restored dir of graphscope-store backup files.
  std::string checkpoint_restore_dir = "/tmp/maxgraph_data/restored";

private:
  GSOptions() = default;
  bool SetOptions(const YAML::Node& map);
};

inline
GSOptions& GSOptions::GetInstance() {
  static GSOptions gs_options;
  return gs_options;
}

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs

#endif // GRAPHSCOPE_LOADER_GS_OPTION_H_
