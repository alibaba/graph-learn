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

#ifndef DGS_COMMON_OPTIONS_H_
#define DGS_COMMON_OPTIONS_H_

#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "generated/proto/coordinator.pb.h"

namespace dgs {

struct RdbEnvOptions {
  uint32_t low_prio_bg_threads_num = 1;
  uint32_t high_prio_bg_threads_num = 1;

  bool SetOptions(const YAML::Node& map);
};

struct SampleStoreOptions {
  /// Specify whether to open in-memory mode of rocksdb.
  bool in_memory_mode = false;

  /// Path to store db.
  std::string db_path = "./";

  /// Path to store backup.
  std::string backup_path = "./";

  std::string memtable_rep = "hashskiplist";

  uint32_t hash_bucket_count = 1024 * 1024;

  uint32_t skip_list_lookahead = 0;

  // default: 64MB
  uint64_t block_cache_capacity = 1UL << 26;

  /// Specify ttl hours for expiring edge data.
  uint32_t ttl_in_hours = 1200;

  bool SetOptions(const YAML::Node& map);
};

struct SubscriptionTableOptions {
  /// Path to store table.
  std::string table_path = "./";

  /// Path to store backup.
  std::string backup_path = "./";

  std::string memtable_rep = "hashskiplist";

  uint32_t hash_bucket_count = 1024 * 1024;

  uint32_t skip_list_lookahead = 0;

  // default: 64MB
  uint64_t block_cache_capacity = 1UL << 26;

  /// Specify ttl hours for expiring subscription rules.
  uint32_t ttl_in_hours = 1200;

  bool SetOptions(const YAML::Node& map);
};

struct RecordPollingOptions {
  /// Specify the kafka servers of polling source.
  std::vector<std::string> source_kafka_servers = { "localhost:9092" };

  std::string FormatKafkaServers() const;

  /// Specify the kafka topic of polling source.
  std::string kafka_topic = "record-polling";

  /// Specify the kafka partition num of polling source.
  uint32_t kafka_partition_num = 4;

  /// Specify the number of threads used for polling.
  uint32_t thread_num = 2;

  /// Specify the time interval in milliseconds for next retrying if no data
  /// is pulled in one attempt.
  uint32_t retry_interval_in_ms = 1000;

  /// Specify the maximum number of polled record batches that downstream
  /// consumers can process in parallel.
  uint32_t process_concurrency = 10;

  bool SetOptions(const YAML::Node& map);
};

struct SamplePublishingOptions {
  /// Specify the kafka servers of output sampled results.
  std::vector<std::string> output_kafka_servers = { "localhost:9092" };

  std::string FormatKafkaServers() const;

  /// Specify the kafka topic of output sampled results.
  std::string kafka_topic = "sample-publishing";

  /// Specify the kafka partition num of output sampled results.
  uint32_t kafka_partition_num = 4;

  /// Specify the size of kafka producer pool.
  uint32_t producer_pool_size = 1;

  /// Specify the maximum retry times of producing.
  uint32_t max_produce_retry_times = 3;

  /// Specify the time interval of polling producing-callbacks.
  uint32_t cb_poll_interval_in_ms = 100;

  bool SetOptions(const YAML::Node& map);
};

struct LoggingOptions {
  /// Specify how many graph updates should be processed between two data logs
  /// produced by sampling actor.
  uint32_t data_log_period = 1;

  /// Specify how many rules should be processed between two data logs produced
  /// by sampling actor.
  uint32_t rule_log_period = 1;

  /// Interval for logging incoming serving requests statistics.
  uint32_t request_log_period = 1;

  bool SetOptions(const YAML::Node& map);
};

struct EventHandlerOptions {
  /// Specify the http port for event serving.
  uint16_t http_port = 10000;

  uint32_t max_local_requests = 6;

  bool SetOptions(const YAML::Node& map);
};

struct CoordinatorClientOptions {
  /// Server ip address
  std::string server_ipaddr = "0.0.0.0:50051";
  uint32_t wait_time_in_sec = 60;
  uint32_t heartbeat_interval_in_sec = 60;

  bool SetOptions(const YAML::Node& map);
};

class Server;

class Options {
public:
  static Options& GetInstance() {
    static Options options;
    return options;
  }

  bool Load(const std::string& yaml_str);
  bool LoadFile(const std::string& option_file);

  WorkerType GetWorkerType() const {
    return worker_type_;
  }

  const std::string& GetFbsFileDir() const {
      return fbs_file_dir;
  }

  const std::string& GetSchemaFile() const {
      return schema_file;
  }

  const RdbEnvOptions& GetRdbEnvOptions() const {
    return rdb_env_options_;
  }

  const SampleStoreOptions& GetSampleStoreOptions() const {
    return sample_store_options_;
  }

  const SubscriptionTableOptions& GetSubscriptionTableOptions() const {
    return subs_table_options_;
  }

  const RecordPollingOptions& GetRecordPollingOptions() const {
    return record_polling_options_;
  }

  const SamplePublishingOptions& GetSamplePublishingOptions() const {
    return sample_pub_options_;
  }

  const LoggingOptions& GetLoggingOptions() const {
    return logging_options_;
  }

  const EventHandlerOptions& GetEventHandlerOptions() const {
    return event_hdl_options_;
  }

  const CoordinatorClientOptions& GetCoordClientOptions() const {
    return coord_client_options_;
  }

private:
  bool SetOptions(const YAML::Node& map);

private:
  /// Specify the type of current worker.
  WorkerType worker_type_ = WorkerType::Sampling;

  /// Specify the directory which contains all .fbs files
  std::string fbs_file_dir = "../../fbs";

  /// Specify the json file of graph schema
  std::string schema_file = "../../conf/schema.template.json";

  /// Compound options
  RdbEnvOptions            rdb_env_options_;
  SampleStoreOptions       sample_store_options_;
  SubscriptionTableOptions subs_table_options_;
  RecordPollingOptions     record_polling_options_;
  SamplePublishingOptions  sample_pub_options_;
  LoggingOptions           logging_options_;
  EventHandlerOptions      event_hdl_options_;
  CoordinatorClientOptions coord_client_options_;

  friend class Server;
};

}  // namespace dgs

#endif  // DGS_COMMON_OPTIONS_H_
