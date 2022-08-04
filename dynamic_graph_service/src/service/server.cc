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

#include "service/server.h"

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include "seastar/core/alien.hh"
#include "seastar/core/when_all.hh"

#include "service/serving_group.actg.h"

namespace dgs {

act::BytesBuffer MakeInstallQueryBuf(const std::string& query_plan);

template <typename Ref, typename... Scopes>
void BuildActorRefs(std::vector<Ref>& refs,  // NOLINT
    std::function<Ref(hiactor::scope_builder&)> create_func,
    Scopes... scopes) {  // NOLINT
  uint32_t local_shard_num = act::LocalShardCount();
  refs.reserve(local_shard_num);
  hiactor::scope_builder builder(0, scopes...);

  for (unsigned i = 0; i < local_shard_num; i++) {
    auto g_sid = act::GlobalShardIdAnchor() + i;
    builder.set_shard(g_sid);
    refs.emplace_back(create_func(builder));
  }
}

Server::Server()
  : actor_system_(nullptr),
    sample_store_(nullptr),
    poller_manager_(),
    this_worker_id_(0),
    rdb_env_(storage::RdbEnv::Default()) {
}

void Server::Init(const InitInfo& info) {
  auto& opts = Options::GetInstance();

  auto& rdb_opts = opts.GetRdbEnvOptions();
  rdb_env_->SetBackgroundThreads(
      static_cast<int>(rdb_opts.high_prio_bg_threads_num),
      storage::RdbEnv::Priority::HIGH);
  rdb_env_->SetBackgroundThreads(
      static_cast<int>(rdb_opts.high_prio_bg_threads_num),
      storage::RdbEnv::Priority::LOW);

  this_worker_id_ = info.worker_id;
  // Init kafka infos of record polling && sample publishing
  auto& us_info = info.upstream_info;
  auto& polling_opts = opts.record_polling_options_;
  polling_opts.source_kafka_servers = us_info->sub_kafka_servers;
  polling_opts.kafka_topic = us_info->sub_kafka_topic;
  polling_opts.kafka_partition_num = us_info->sub_kafka_partition_num;

  auto& ds_kafka_info = info.downstream_kafka_info;
  if (ds_kafka_info) {
    auto& publishing_opts = opts.sample_pub_options_;
    publishing_opts.output_kafka_servers = ds_kafka_info->pub_kafka_servers;
    publishing_opts.kafka_topic = ds_kafka_info->pub_kafka_topic;
    publishing_opts.kafka_partition_num = ds_kafka_info->pub_kafka_partition_num;
  }

  auto& store_partition_info = info.store_partition_info;
  auto& checkpoint_info = info.checkpoint_info;

  // Init sample store
  auto& store_opt = Options::GetInstance().GetSampleStoreOptions();
  try {
    auto partitioner = PartitionerFactory::Create(
        store_partition_info.partition_strategy,
        store_partition_info.partition_num);
    sample_store_ = std::make_unique<storage::SampleStore>(
        checkpoint_info.sample_store_backup_infos, std::move(partitioner),
        store_opt.db_path, store_opt.backup_path, rdb_env_);
    LOG(INFO) << "SampleStore is created with partitioning strategy "
              << store_partition_info.partition_strategy
              << ", partition num is "
              << store_partition_info.partition_num;
  } catch (std::exception& ex) {
    LOG(FATAL) << "Creating SampleStore failed: " << ex.what() << std::endl;
  }

  actor_system_ = std::make_unique<ActorSystem>(
      info.worker_type, info.worker_id,
      info.num_workers, info.num_local_actor_shards);
  LOG(INFO) << "ActorSystem is running now.";

  // Init record poller manager
  poller_manager_.Init(
      PartitionerFactory::Create(
          store_partition_info.partition_strategy,
          store_partition_info.partition_num),
      store_partition_info.routing_info,
      us_info->sub_kafka_pids,
      checkpoint_info.ingested_offsets);
  LOG(INFO) << "LogPollingManager is created.";
}

void Server::Finalize() {
  poller_manager_.Stop();
  LOG(INFO) << "Record Poller is stopped.";
}

void Server::Start() {
  poller_manager_.Start();
  LOG(INFO) << "Record Poller is started.";
}

void Server::Backup(
    std::vector<storage::StorePartitionBackupInfo>* sample_store_backup_infos,
    std::vector<storage::SubsPartitionBackupInfo>* subs_table_backup_infos) {
  *sample_store_backup_infos = sample_store_->Backup();
}

SamplingServer::SamplingServer()
  : Server(), sample_builder_(),
    subs_table_(), sampling_refs_() {}

void SamplingServer::Init(const InitInfo& init_info) {
  Server::Init(init_info);

  // Init kafka producer pool
  KafkaProducerPool::GetInstance()->Init();
  LOG(INFO) << "Kafka Producer Pool is inited.";

  // Create actor refs.
  BuildActorRefs<SamplingActor_ref>(sampling_refs_,
      MakeSamplingActorInstRef);

  auto& store_part_info = init_info.store_partition_info;
  auto& checkpoint_info = init_info.checkpoint_info;

  // Create Sample builder.
  sample_builder_ = std::make_unique<storage::SampleBuilder>(
      store_part_info.managed_pids,
      PartitionerFactory::Create(
          store_part_info.partition_strategy,
          store_part_info.partition_num));

  auto& opt = Options::GetInstance().GetSubscriptionTableOptions();
  // Create SubscriptionTable.
  auto partitioner = PartitionerFactory::Create(
      store_part_info.partition_strategy,
      store_part_info.partition_num);
  subs_table_ = std::make_unique<storage::SubscriptionTable>(
    checkpoint_info.subs_table_backup_infos, std::move(partitioner),
    opt.table_path, opt.backup_path, rdb_env_);

  auto& ds_partition_info = init_info.downstream_partition_info;
  subs_table_->SetDSWorkerPartitioner(
      ds_partition_info->worker_partition_num,
      ds_partition_info->worker_partition_strategy);

  auto buf = MakeInstallQueryBuf(init_info.query_plan);
  auto payload = std::make_shared<SamplingInitPayload>(
      std::move(buf), sample_store_.get(),
      sample_builder_.get(), subs_table_.get(),
      init_info.store_partition_info.partition_strategy,
      init_info.store_partition_info.partition_num,
      init_info.store_partition_info.routing_info,
      init_info.downstream_partition_info->worker_partition_num,
      init_info.downstream_partition_info->kafka_to_wid);

  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0,
      [this, payload] () mutable {
    uint32_t count = act::LocalShardCount();
    return seastar::parallel_for_each(boost::irange(0u, count),
        [this, payload] (uint32_t i) {
      return sampling_refs_[i].ExecuteAdminOperation(
          AdminRequest(AdminOperation::INIT, payload)).discard_result();
    });
  });

  fut.wait();
  LOG(INFO) << "Sampling actors are inited.";
}

void SamplingServer::Finalize() {
  Server::Finalize();
  KafkaProducerPool::GetInstance()->Finalize();
  LOG(INFO) << "Kafka Producer Pool is deleted.";
}

void SamplingServer::Backup(
    std::vector<storage::StorePartitionBackupInfo>* sample_store_backup_infos,
    std::vector<storage::SubsPartitionBackupInfo>* subs_table_backup_infos) {
  *sample_store_backup_infos = sample_store_->Backup();
  *subs_table_backup_infos = subs_table_->Backup();
}

ServingServer::ServingServer()
  : Server(), serving_refs_(),
    data_update_refs_(),
    event_handler_(nullptr) {
}

void ServingServer::Init(const InitInfo& init_info) {
  Server::Init(init_info);

  // Create actor refs.
  BuildActorRefs<ServingActor_ref>(serving_refs_,
      MakeServingActorInstRef, MakeServingGroupScope());
  BuildActorRefs<DataUpdateActor_ref>(data_update_refs_,
      MakeDataUpdateActorInstRef, MakeServingGroupScope());

  auto buf = MakeInstallQueryBuf(init_info.query_plan);
  auto payload = std::make_shared<ServingInitPayload>(
      std::move(buf), sample_store_.get());

  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0,
      [this, payload] () mutable {
    uint32_t count = act::LocalShardCount();
    return seastar::parallel_for_each(boost::irange(0u, count),
        [this, payload] (uint32_t i) {
      return seastar::when_all(
        serving_refs_[i].ExecuteAdminOperation(
          AdminRequest(AdminOperation::INIT, payload)).discard_result(),
        data_update_refs_[i].ExecuteAdminOperation(
          AdminRequest(AdminOperation::INIT, payload)).discard_result()
      ).discard_result();  // NOLINT
    });
  });  // NOLINT
  fut.wait();
  LOG(INFO) << "Serving actors are inited.";
}

void ServingServer::Finalize() {
  if (event_handler_) {
    auto fut = seastar::alien::submit_to(
        *seastar::alien::internal::default_instance, 0,
        [this] { return event_handler_->Stop(); });
    fut.wait();
    LOG(INFO) << "Event handler is stopped";
  }

  if (rate_limiter_) {
    rate_limiter_->Stop();
  }
  LOG(INFO) << "Adaptive rate limiter is stopped.";

  Server::Finalize();
}

void ServingServer::Start() {
  Server::Start();

  rate_limiter_ = std::make_unique<AdaptiveRateLimiter>(&poller_manager_);
  rate_limiter_->Start();
  LOG(INFO) << "Adaptive rate limiter is started.";

  auto http_port = Options::GetInstance().GetEventHandlerOptions().http_port;
  event_handler_ = std::make_unique<EventHandler>(
      this_worker_id_, http_port, rate_limiter_.get());

  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0,
      [this] { return event_handler_->Start(); });
  fut.wait();
  LOG(INFO) << "Event handler is started.";
}

Server::UpstreamInfo::UpstreamInfo(
    std::vector<std::string>&& sub_kafka_servers,
    const std::string& sub_kafka_topic,
    uint32_t sub_kafka_partition_num,
    std::vector<PartitionId>&& sub_kafka_pids)
  : sub_kafka_servers(std::move(sub_kafka_servers)),
    sub_kafka_topic(sub_kafka_topic),
    sub_kafka_partition_num(sub_kafka_partition_num),
    sub_kafka_pids(std::move(sub_kafka_pids)) {
}

Server::DownstreamKafkaInfo::DownstreamKafkaInfo(
    std::vector<std::string>&& pub_kafka_servers,
    const std::string& pub_kafka_topic,
    uint32_t pub_kafka_partition_num)
  : pub_kafka_servers(std::move(pub_kafka_servers)),
    pub_kafka_topic(pub_kafka_topic),
    pub_kafka_partition_num(pub_kafka_partition_num) {
}

Server::DownstreamPartitionInfo::DownstreamPartitionInfo(
    const std::string& worker_partition_strategy,
    uint32_t worker_partition_num,
    std::vector<uint32_t>&& kafka_to_wid)
  : worker_partition_strategy(worker_partition_strategy),
    worker_partition_num(worker_partition_num),
    kafka_to_wid(std::move(kafka_to_wid)) {
}

Server::StorePartitionInfo::StorePartitionInfo(
    const std::string& partition_strategy,
    uint32_t partition_num,
    std::vector<PartitionId>&& managed_pids,
    std::vector<ShardId>&& routing_info)
  : partition_strategy(partition_strategy),
    partition_num(partition_num),
    managed_pids(std::move(managed_pids)),
    routing_info(std::move(routing_info)) {
}

Server::CheckpointInfo::CheckpointInfo(
    std::unordered_map<uint32_t, int64_t>&& ingested_offsets,
    std::vector<storage::StorePartitionBackupInfo>&& sample_store_backup_infos,
    std::vector<storage::SubsPartitionBackupInfo>&& subs_table_backup_infos)
  : ingested_offsets(std::move(ingested_offsets)),
    sample_store_backup_infos(std::move(sample_store_backup_infos)),
    subs_table_backup_infos(std::move(subs_table_backup_infos)) {
}

Server::InitInfo::InitInfo(
    WorkerType worker_type,
    uint32_t worker_id,
    uint32_t num_workers,
    const std::string& query_plan,
    uint32_t num_local_actor_shards,
    Server::StorePartitionInfo&& store_partition_info,
    Server::CheckpointInfo&& checkpoint_info,
    std::unique_ptr<UpstreamInfo>&& upstream_info,
    std::unique_ptr<DownstreamKafkaInfo>&& ds_kafka_info,
    std::unique_ptr<DownstreamPartitionInfo>&& ds_partition_info)
  : worker_type(worker_type),
    worker_id(worker_id),
    num_workers(num_workers),
    query_plan(query_plan),
    num_local_actor_shards(num_local_actor_shards),
    store_partition_info(std::move(store_partition_info)),
    checkpoint_info(std::move(checkpoint_info)),
    upstream_info(std::move(upstream_info)),
    downstream_kafka_info(std::move(ds_kafka_info)),
    downstream_partition_info(std::move(ds_partition_info)) {
}

act::BytesBuffer MakeInstallQueryBuf(const std::string& query_plan) {
  auto& opts = Options::GetInstance();
  std::string schema_str;
  std::string schema_file = opts.GetFbsFileDir() + "/install_query_req.fbs";

  bool ok = flatbuffers::LoadFile(schema_file.c_str(), false, &schema_str);
  if (!ok) {
    LOG(ERROR) << "Load install_query_request schema file failed.\n";
  }

  flatbuffers::Parser parser;

  const char* include_paths[] = { opts.GetFbsFileDir().c_str() };
  ok = parser.Parse(schema_str.c_str(), include_paths);
  if (!ok) {
    LOG(FATAL) << "Parse install_query_request schema file failed.\n";
  }
  ok = parser.Parse(query_plan.c_str());
  if (!ok) {
    LOG(FATAL) << "Parse query plan json file failed.\n";
  }

  auto* ptr = reinterpret_cast<char*>(parser.builder_.GetBufferPointer());
  auto size = parser.builder_.GetSize();
  return {ptr, size};
}

}  // namespace dgs
