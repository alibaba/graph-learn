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

#include "common/options.h"

#include "common/log.h"

namespace dgs {

template <typename T>
bool SetScalarOption(T& target, const YAML::Node& map,  // NOLINT
                     const std::string& group, const std::string& option) {
  auto node = map[option];
  if (node) {
    if (!node.IsScalar()) {
      LOG(ERROR) << "Wrong type with input " << group << ":" << option
                 << ", scalar type required.";
      return false;
    }
    try {
      target = node.as<T>();
    } catch (const YAML::TypedBadConversion<T>& e) {
      LOG(ERROR) << "Type conversion error with input "
                 << group << ":" << option << ", " << e.what();
      return false;
    }
  }
  return true;
}

template <typename T>
bool SetListOption(std::vector<T>& target, const YAML::Node& map,  // NOLINT
                   const std::string& group, const std::string& option) {
  auto node = map[option];
  if (node) {
    if (!node.IsSequence()) {
      LOG(ERROR) << "Wrong type with input " << group << ":" << option
                 << ", sequence type required.";
      return false;
    }
    std::vector<T> list(node.size());
    for (size_t i = 0; i < node.size(); i++) {
      try {
        list[i] = node[i].as<T>();
      } catch (const YAML::TypedBadConversion<T>& e) {
        LOG(ERROR) << "Type conversion error with input "
                   << group << ":" << option << ", " << e.what();
        return false;
      }
    }
    target = std::move(list);
  }
  return true;
}

std::string StrJoin(const std::vector<std::string>& lists,
                    const std::string& delim) {
  if (lists.empty()) {
    return "";
  }
  std::string join = lists[0];
  for (size_t i = 1; i < lists.size(); i++) {
    join += delim;
    join += lists[i];
  }
  return join;
}

bool RdbEnvOptions::SetOptions(const YAML::Node& map) {
  if (!map) {
    return true;
  }
  std::string group = "rdb-env-option";
  bool suc = true;
  suc &= SetScalarOption(high_prio_bg_threads_num, map, group,
      "high-prio-bg-threads-num");
  suc &= SetScalarOption(low_prio_bg_threads_num, map, group,
      "low-prio-bg-threads-num");
  return suc;
}

bool SampleStoreOptions::SetOptions(const YAML::Node& map) {
  if (!map) {
    return true;
  }
  std::string group = "sample-store-option";
  bool suc = true;
  suc &= SetScalarOption(
      in_memory_mode, map, group, "in-memory-mode");
  suc &= SetScalarOption(
      db_path, map, group, "db-path");
  suc &= SetScalarOption(
      backup_path, map, group, "backup-path");
  suc &= SetScalarOption(
      memtable_rep, map, group, "memtable-rep");
  suc &= SetScalarOption(
      hash_bucket_count, map, group, "hash-bucket-count");
  suc &= SetScalarOption(
      skip_list_lookahead, map, group, "skip-list-lookahead");
  suc &= SetScalarOption(
      block_cache_capacity, map, group, "block-cache-capacity");
  suc &= SetScalarOption(
      ttl_in_hours, map, group, "ttl-hours");
  return suc;
}

bool SubscriptionTableOptions::SetOptions(const YAML::Node& map) {
  if (!map) {
    return true;
  }
  std::string group = "subscription-table-option";
  bool suc = true;
  suc &= SetScalarOption(table_path, map, group, "table-path");
  suc &= SetScalarOption(backup_path, map, group, "backup-path");
  suc &= SetScalarOption(memtable_rep, map, group, "memtable-rep");
  suc &= SetScalarOption(hash_bucket_count, map, group, "hash-bucket-count");
  suc &= SetScalarOption(skip_list_lookahead, map, group,
                         "skip-list-lookahead");
  suc &= SetScalarOption(block_cache_capacity, map, group,
                         "block-cache-capacity");
  suc &= SetScalarOption(ttl_in_hours, map, group, "ttl-hours");
  return suc;
}

bool RecordPollingOptions::SetOptions(const YAML::Node& map) {
  if (!map) {
    return true;
  }
  std::string group = "record-polling-option";
  bool suc = true;
  suc &= SetListOption(
      source_kafka_servers, map, group, "source-kafka-servers");
  suc &= SetScalarOption(kafka_topic, map, group, "kafka-topic");
  suc &= SetScalarOption(
      kafka_partition_num, map, group, "kafka-partition-num");
  suc &= SetScalarOption(thread_num, map, group, "thread-num");
  suc &= SetScalarOption(
      retry_interval_in_ms, map, group, "retry-interval-ms");
  suc &= SetScalarOption(
      process_concurrency, map, group, "process-concurrency");
  return suc;
}

std::string RecordPollingOptions::FormatKafkaServers() const {
  return StrJoin(source_kafka_servers, ",");
}

bool SamplePublishingOptions::SetOptions(const YAML::Node& map) {
  if (!map) {
    return true;
  }
  std::string group = "sample-publishing-option";
  bool suc = true;
  suc &= SetListOption(output_kafka_servers, map, group,
                       "output-kafka-servers");
  suc &= SetScalarOption(kafka_topic, map, group, "kafka-topic");
  suc &= SetScalarOption(
    kafka_partition_num, map, group, "kafka-partition-num");
  suc &= SetScalarOption(
    producer_pool_size, map, group, "producer-pool-size");
  suc &= SetScalarOption(
    max_produce_retry_times, map, group, "max-produce-retry-times");
  suc &= SetScalarOption(cb_poll_interval_in_ms, map, group,
                         "callback-poll-interval-ms");
  return suc;
}

std::string SamplePublishingOptions::FormatKafkaServers() const {
  return StrJoin(output_kafka_servers, ",");
}

bool LoggingOptions::SetOptions(const YAML::Node& map) {
  if (!map) {
    return true;
  }
  std::string group = "logging-option";
  bool suc = true;
  suc &= SetScalarOption(data_log_period, map, group, "data-log-period");
  suc &= SetScalarOption(rule_log_period, map, group, "rule-log-period");
  suc &= SetScalarOption(request_log_period, map, group, "request-log-period");
  return suc;
}

bool EventHandlerOptions::SetOptions(const YAML::Node& map) {
  if (!map) {
    return true;
  }
  std::string group = "event-handler-option";
  bool suc = true;
  suc &= SetScalarOption(http_port, map, group, "http-port");
  suc &= SetScalarOption(http_port, map, group, "max-local-requests");
  return suc;
}

bool CoordinatorClientOptions::SetOptions(const YAML::Node& map) {
  if (!map) {
    return true;
  }
  std::string group = "coordinator-client-option";
  bool ok = true;
  ok &= SetScalarOption(server_ipaddr, map, group, "server-ipaddr");
  ok &= SetScalarOption(wait_time_in_sec, map, group, "wait-time-in-sec");
  ok &= SetScalarOption(heartbeat_interval_in_sec, map, group,
                        "heartbeat-interval-in-sec");
  return ok;
}

bool Options::Load(const std::string& yaml_str) {
  YAML::Node node;
  try {
    node = YAML::Load(yaml_str);
  } catch (const YAML::ParserException& e) {
    LOG(ERROR) << "Error in loading option yaml string:" << e.what();
    return false;
  }
  return SetOptions(node);
}

bool Options::LoadFile(const std::string& option_file) {
  YAML::Node node;
  try {
    node = YAML::LoadFile(option_file);
  } catch (const YAML::BadFile& e) {
    LOG(ERROR) << "Error in loading option file: " << e.what();
    return false;
  }
  return SetOptions(node);
}

bool Options::SetOptions(const YAML::Node& map) {
  std::string worker_type_str;
  if (!SetScalarOption(worker_type_str, map, "option", "worker-type")) {
    return false;
  }
  if (!worker_type_str.empty()) {
    if (worker_type_str == "Sampling") {
      worker_type_ = WorkerType::Sampling;
    } else if (worker_type_str == "Serving") {
      worker_type_ = WorkerType::Serving;
    } else {
      LOG(ERROR) << "Set program worker type with unknown option: "
                 << worker_type_str;
      return false;
    }
  }
  bool suc = true;
  suc &= SetScalarOption(fbs_file_dir, map, "option", "fbs-file-dir");
  suc &= SetScalarOption(schema_file, map, "option", "schema-file");
  suc &= rdb_env_options_.SetOptions(map["rdb-env"]);
  suc &= sample_store_options_.SetOptions(map["sample-store"]);
  suc &= subs_table_options_.SetOptions(map["subscription-table"]);
  suc &= record_polling_options_.SetOptions(map["record-polling"]);
  suc &= sample_pub_options_.SetOptions(map["sample-publishing"]);
  suc &= logging_options_.SetOptions(map["logging"]);
  suc &= event_hdl_options_.SetOptions(map["event-handler"]);
  suc &= coord_client_options_.SetOptions(map["coordinator-client"]);
  return suc;
}

}  // namespace dgs
