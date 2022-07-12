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

#ifndef DGS_SERVICE_SERVER_H_
#define DGS_SERVICE_SERVER_H_

#include "common/options.h"
#include "core/storage/sample_builder.h"
#include "core/storage/sample_store.h"
#include "core/storage/subscription_table.h"
#include "service/actor_system.h"
#include "service/actor_ref_builder.h"
#include "service/adaptive_rate_limiter.h"
#include "service/channel/record_poller.h"
#include "service/event_handler.h"

namespace dgs {

class Server {
public:
  struct UpstreamInfo;
  struct DownstreamKafkaInfo;
  struct DownstreamWorkerPartitionInfo;
  struct StorePartitionInfo;
  struct CheckpointInfo;
  struct InitInfo;

public:
  Server();
  virtual ~Server() = default;

  virtual void Init(const InitInfo& init_info);
  virtual void Finalize();

  virtual void Start();

  virtual void BlockUntilReady();

  virtual void Pause() = 0;
  virtual void Resume() = 0;

  virtual bool AddStoragePartition() { return true; }
  virtual bool RemoveStoragePartition() { return true; }
  virtual bool BackupStoragePartition() { return true; }

  std::unordered_map<uint32_t, int64_t> GetIngestedOffsets() {
    return poller_manager_.GetIngestedOffsets();
  }

  virtual void Backup(
    std::vector<storage::StorePartitionBackupInfo>* sample_store_backup_infos,
    std::vector<storage::SubsPartitionBackupInfo>* subs_table_backup_infos);

protected:
  std::unique_ptr<ActorSystem>          actor_system_;
  std::unique_ptr<storage::SampleStore> sample_store_;
  RecordPollingManager                  poller_manager_;
  WorkerId                              this_worker_id_;
  storage::RdbEnv*                      rdb_env_;
};

class SamplingServer final : public Server {
public:
  SamplingServer();
  ~SamplingServer() final = default;

  void Init(const InitInfo& init_info) final;
  void Finalize() final;

  void Pause() final {}
  void Resume() final {}

  void Backup(
      std::vector<storage::StorePartitionBackupInfo>* sample_store_backup_infos,
      std::vector<storage::SubsPartitionBackupInfo>* subs_table_backup_infos) final;

private:
  std::unique_ptr<storage::SampleBuilder>     sample_builder_;
  std::unique_ptr<storage::SubscriptionTable> subs_table_;
  std::vector<SamplingActor_ref>              sampling_refs_;
};

class ServingServer final : public Server {
public:
  ServingServer();
  ~ServingServer() final = default;

  void Init(const InitInfo& init_info) final;
  void Finalize() final;

  void BlockUntilReady() final;

  void Pause() final {}
  void Resume() final {}

private:
  std::unique_ptr<EventHandler>        event_handler_;
  std::unique_ptr<AdaptiveRateLimiter> rate_limiter_;
  std::vector<ServingActor_ref>        serving_refs_;
  std::vector<DataUpdateActor_ref>     data_update_refs_;
};

struct Server::UpstreamInfo {
  std::vector<std::string> sub_kafka_servers;
  std::string sub_kafka_topic;
  uint32_t sub_kafka_partition_num;
  std::vector<PartitionId> sub_kafka_pids;

  UpstreamInfo(std::vector<std::string>&& sub_kafka_servers,
               const std::string& sub_kafka_topic,
               uint32_t sub_kafka_partition_num,
               std::vector<PartitionId>&& sub_kafka_pids);
};

struct Server::DownstreamKafkaInfo {
  std::vector<std::string> pub_kafka_servers;
  std::string pub_kafka_topic;
  uint32_t pub_kafka_partition_num;

  DownstreamKafkaInfo(std::vector<std::string>&& pub_kafka_servers,
                      const std::string& pub_kafka_topic,
                      uint32_t pub_kafka_partition_num);
};

struct Server::DownstreamWorkerPartitionInfo {
  std::string worker_partition_strategy;
  uint32_t    worker_partition_num;
  std::vector<uint32_t> kafka_to_worker_pid_vec;

  DownstreamWorkerPartitionInfo(const std::string& worker_partition_strategy,
                                uint32_t worker_partition_num,
                                std::vector<uint32_t>&& kafka_to_worker_pid_vec);
};

struct Server::StorePartitionInfo {
  std::string partition_strategy = "hash";
  uint32_t    partition_num = 1;
  std::vector<PartitionId> managed_pids;
  std::vector<ShardId>     routing_info;

  StorePartitionInfo() = default;
  StorePartitionInfo(const std::string& partition_strategy,
                     uint32_t partition_num,
                     std::vector<PartitionId>&& managed_pids,
                     std::vector<ShardId>&& routing_info);
  StorePartitionInfo(StorePartitionInfo&&) = default;
  StorePartitionInfo& operator=(StorePartitionInfo&&) = default;
};

struct Server::CheckpointInfo {
  std::unordered_map<uint32_t, int64_t> ingested_offsets;
  std::vector<storage::StorePartitionBackupInfo> sample_store_backup_infos;
  std::vector<storage::SubsPartitionBackupInfo>  subs_table_backup_infos;

  CheckpointInfo() = default;
  CheckpointInfo(
    std::unordered_map<uint32_t, int64_t>&& ingested_offsets,
    std::vector<storage::StorePartitionBackupInfo>&& sample_store_backup_infos,
    std::vector<storage::SubsPartitionBackupInfo>&& subs_table_backup_infos);
  CheckpointInfo(CheckpointInfo&&) = default;
  CheckpointInfo& operator=(CheckpointInfo&&) = default;
};

struct Server::InitInfo {
  WorkerType  worker_type;
  uint32_t    worker_id;
  uint32_t    num_workers;
  std::string query_plan;
  uint32_t    num_local_actor_shards;
  StorePartitionInfo              store_partition_info;
  CheckpointInfo                  checkpoint_info;
  std::unique_ptr<UpstreamInfo>   upstream_info;
  std::unique_ptr<DownstreamKafkaInfo> downstream_kafka_info;
  std::unique_ptr<DownstreamWorkerPartitionInfo> downstream_partition_info;

  InitInfo(WorkerType worker_type,
           uint32_t worker_id,
           uint32_t num_workers,
           const std::string& query_plan,
           uint32_t num_local_actor_shards,
           StorePartitionInfo&& store_partition_info,
           CheckpointInfo&& checkpoint_info,
           std::unique_ptr<UpstreamInfo>&& upstream_info,
           std::unique_ptr<DownstreamKafkaInfo>&& ds_kafka_info,
           std::unique_ptr<DownstreamWorkerPartitionInfo>&& ds_partition_info);
};

}  // namespace dgs

#endif  // DGS_SERVICE_SERVER_H_
