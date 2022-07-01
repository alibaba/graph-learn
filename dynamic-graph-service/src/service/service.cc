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

#include "service/service.h"

#include <fstream>

#include "grpc/support/time.h"
#include "grpcpp/create_channel.h"

#include "seastar/core/alien.hh"

#include "common/actor_wrapper.h"
#include "common/host.h"
#include "common/log.h"
#include "common/options.h"

namespace dgs {

void WriteServerConfigToTmpFile(uint32_t worker_id,
                                const GetInitInfoResponsePb* res);
std::unique_ptr<Coordinator::Stub> CreateCoordinatorStub();

Service::Service(const std::string& config_file, uint32_t worker_id)
  : worker_id_(worker_id), server_(nullptr), is_termination_(false) {
  // init logging system.
  InitGoogleLogging();

  Configure(config_file);

  if (worker_type_ == WorkerType::Sampling) {
    server_ = std::make_unique<SamplingServer>();
  } else if (worker_type_ == WorkerType::Serving) {
    server_ = std::make_unique<ServingServer>();
  } else {
    throw std::runtime_error("Unsupported worker type for service.");
  }

  stub_ = CreateCoordinatorStub();
}

Service::~Service() {
  UninitGoogleLogging();
}

void Service::Run() {
  RegisterSelf();

  std::thread reporter;

  std::unique_ptr<Server::InitInfo> init_info;
  RetrieveInitInfo(&init_info);
  if (is_termination_.load(std::memory_order_relaxed)) {
      goto FINAL;
  }
  server_->Init(*init_info);

  WaitUntilAllAreInited();
  if (is_termination_.load(std::memory_order_relaxed)) {
    goto FINAL;
  }

  server_->Start();

  reporter = std::thread(&Service::ReportStatsInfo, this);

  WaitUntilAllAreReady();
  if (is_termination_.load(std::memory_order_relaxed)) {
    goto FINAL;
  }

  server_->BlockUntilReady();
  LOG(INFO) << "Service is ready...";

  ReportSelfIsReady();

FINAL:
  if (reporter.joinable()) {
    reporter.join();
  }
  server_->Finalize();
  LOG(INFO) << "Service is terminated...";
}

void Service::RegisterSelf() {
  grpc::ClientContext context;
  RegisterWorkerRequestPb  req;
  RegisterWorkerResponsePb res;

  auto ipaddr = GetLocalEndpoint(GetAvailablePort());
  req.set_worker_ip(ipaddr);
  req.set_worker_id(worker_id_);
  req.set_worker_type(worker_type_);

  auto s = stub_->RegisterWorker(&context, req, &res);
  if (!s.ok()) {
    throw std::runtime_error(
        "Failed to register current worker: " + s.error_message());
  }
  if (!res.suc()) {
    throw std::runtime_error(
        "Failed to register current worker with id " + std::to_string(worker_id_));
  }
  num_workers_ = res.num_workers();

  LOG(INFO) << "Registered current worker with id: " << worker_id_;
}

void Service::WaitUntilAllAreInited() {
  grpc::ClientContext context;
  ReportInitedRequestPb req;
  ReportInitedResponsePb res;

  req.set_worker_id(worker_id_);
  req.set_worker_type(worker_type_);

  auto s = stub_->ReportInited(&context, req, &res);
  if (!s.ok()) {
    throw std::runtime_error(s.error_message());
  }

  if (res.terminate_service()) {
    is_termination_.store(true);
  }
}

void Service::WaitUntilAllAreReady() {
  CheckReadyInfoRequestPb  req;
  CheckReadyInfoResponsePb res;
  grpc::ClientContext context;

  req.set_worker_type(worker_type_);
  req.set_worker_id(worker_id_);

  auto s = stub_->GetCheckReadyInfo(&context, req, &res);
  if (!s.ok()) {
    throw std::runtime_error(s.error_message());
  }

  if (res.terminate_service()) {
    is_termination_.store(true);
  }
}

void Service::ReportSelfIsReady() {
  ServerIsReadyRequestPb  req;
  ServerIsReadyResponsePb res;
  grpc::ClientContext context;

  req.set_worker_type(worker_type_);
  req.set_worker_id(worker_id_);

  auto s = stub_->ReportServerIsReady(&context, req, &res);
  if (!s.ok()) {
    throw std::runtime_error(s.error_message());
  }

  if (res.terminate_service()) {
    is_termination_.store(true);
  }
}

void Service::ReportStatsInfo() {
  ReportStatisticsRequestPb  req;
  ReportStatisticsResponsePb res;
  grpc::Status s;

  req.set_worker_id(worker_id_);
  req.set_worker_type(worker_type_);

  auto heartbeat_interval =
      Options::GetInstance().GetCoordClientOptions().heartbeat_interval_in_sec;
  uint32_t failed_times = 0;

  while (!is_termination_) {
    // report statistics with a fixed interval.
    // the first report will begin after an initial delay.
    std::this_thread::sleep_for(std::chrono::seconds(heartbeat_interval));

    // get ready polling offsets
    auto ingested_offsets = server_->GetIngestedOffsets();
    req.clear_ready_kafka_offsets();
    for (auto& tuple: ingested_offsets) {
      auto* entry = req.add_ready_kafka_offsets();
      entry->set_pid(tuple.first);
      entry->set_ready_offset(tuple.second);
    }

    grpc::ClientContext context;
    s = stub_->ReportStatistics(&context, req, &res);
    if (!s.ok()) {
      failed_times++;
      LOG(WARNING) << "Failed to report statistics by "
                   << failed_times << " times";
      if (failed_times >= 3) {
        LOG(WARNING) << "Start to terminate service ...";
        is_termination_.store(true);
        break;
      }
    } else {
      failed_times = 0;
      if (res.terminate_service()) {
        is_termination_.store(true);
        break;
      }
      if (res.do_backup()) {
        LOG(INFO) << "Receiving checkpoint request from coordinator, "
                  << "start to do backups ...";
        boost::asio::post(backup_executor_, [this] {
          ReportBackupFinishedRequestPb  req;
          ReportBackupFinishedResponsePb res;
          grpc::ClientContext context;

          req.set_worker_id(worker_id_);
          req.set_worker_type(worker_type_);

          std::vector<storage::StorePartitionBackupInfo> store_backups;
          std::vector<storage::SubsPartitionBackupInfo> subs_backups;
          server_->Backup(&store_backups, &subs_backups);
          for (auto& info : store_backups) {
            auto* entry = req.add_sample_store_backups();
            entry->set_pid(info.pid);
            entry->set_valid(info.valid);
            entry->set_vertex_bid(info.vertex_bid);
            entry->set_edge_bid(info.edge_bid);
          }
          for (auto& info : subs_backups) {
            auto* entry = req.add_subs_table_backups();
            entry->set_pid(info.pid);
            entry->set_valid(info.valid);
            entry->set_bid(info.bid);
          }

          auto s = stub_->ReportBackupFinished(&context, req, &res);
          if (s.ok()) {
            if (res.is_valid()) {
              LOG(INFO) << "Finished backups are synchronized to coordinator";
            } else {
              LOG(WARNING) << "Reported backups are invalid, delete them ...";
              // TODO(@houbai.zzc): delete the invalid backups
            }
          } else {
            LOG(WARNING) << "Failed to report backups to coordinator";
            // TODO(@houbai.zzc): try reporting again
          }
        });
      }
    }
  }

  backup_executor_.stop();
  backup_executor_.join();

  WriteTerminateFlag();
}

void Service::Configure(const std::string& config_file) {
  bool ok = Options::GetInstance().LoadFile(config_file);
  if (!ok) {
    throw std::runtime_error("Cannot open config file: " + config_file);
  }

  ok = Schema::GetInstance().Init();
  if (!ok) {
    throw std::runtime_error("Failed to init graph schema.");
  }

  LOG(INFO) << "Configure service succeed with config file: " << config_file;
  worker_type_ = Options::GetInstance().GetWorkerType();
}

void Service::WriteTerminateFlag() {
  std::ofstream file("./TERMINATE_FLAG");
  file << "TERMINATED";
  file.close();
}

std::unique_ptr<Server::UpstreamInfo>
RetrieveUpstreamInfo(const UpStreamInfoPb& pb) {
  // get kafka servers
  std::vector<std::string> sub_kafka_servers;
  sub_kafka_servers.reserve(pb.sub_kafka_servers_size());
  for (auto& server : pb.sub_kafka_servers()) {
    sub_kafka_servers.push_back(server);
  }
  auto& sub_kafka_topic = pb.sub_kafka_topic();
  uint32_t sub_kafka_partition_num = pb.sub_kafka_partition_num();
  // get subscribed kafka partition ids
  std::vector<PartitionId> sub_kafka_pids;
  sub_kafka_pids.reserve(pb.sub_kafka_pids_size());
  for (auto pid : pb.sub_kafka_pids()) {
    sub_kafka_pids.push_back(pid);
  }

  return std::make_unique<Server::UpstreamInfo>(
      std::move(sub_kafka_servers), sub_kafka_topic,
      sub_kafka_partition_num, std::move(sub_kafka_pids));
}

std::unique_ptr<Server::DownstreamInfo>
RetrieveDownstreamInfo(const DownStreamInfoPb& pb) {
  auto& store_partition_strategy = pb.store_partition_strategy();
  uint32_t store_partition_num = pb.store_partition_num();
  auto& worker_partition_strategy = pb.worker_partition_strategy();
  uint32_t worker_partition_num = pb.worker_partition_num();
  // get kafka servers
  std::vector<std::string> pub_kafka_servers;
  pub_kafka_servers.reserve(pb.pub_kafka_servers_size());
  for (auto& server : pb.pub_kafka_servers()) {
    pub_kafka_servers.push_back(server);
  }
  auto& pub_kafka_topic = pb.pub_kafka_topic();
  uint32_t pub_kafka_partition_num = pb.pub_kafka_partition_num();
  // get published kafka partition ids
  std::vector<PartitionId> pub_kafka_pids;
  pub_kafka_pids.reserve(pb.pub_kafka_pids_size());
  for (auto pid : pb.pub_kafka_pids()) {
    pub_kafka_pids.push_back(pid);
  }

  return std::make_unique<Server::DownstreamInfo>(
      store_partition_strategy, store_partition_num,
      worker_partition_strategy, worker_partition_num,
      std::move(pub_kafka_servers), pub_kafka_topic,
      pub_kafka_partition_num, std::move(pub_kafka_pids));
}

Server::StorePartitionInfo
RetrieveStorePartitionInfo(uint32_t worker_id,
                           WorkerType worker_type,
                           uint32_t actor_lshard_num,
                           const StorePartitionInfoPb& pb) {
  auto& partition_strategy = pb.partition_strategy();
  uint32_t partition_num = pb.partition_num();

  // managed store partition ids.
  std::vector<PartitionId> managed_pids;
  // - size: store_partition_num.
  // - index: store partition id,
  // - value: global actor shard id.
  std::vector<ShardId> routing_info;
  routing_info.resize(partition_num);

  for (int i = 0; i < pb.managed_pids_group_size(); ++i) {
    auto &pid_group = pb.managed_pids_group(i);
    if (pid_group.worker_id() == worker_id) {
      for (int j = 0; j < pid_group.pids_size(); ++j) {
        managed_pids.push_back(pid_group.pids(j));
      }
    }

    if (worker_type == WorkerType::Sampling) {
      const uint32_t gsid_offset = pid_group.worker_id() * actor_lshard_num;
      for (int j = 0; j < pid_group.pids_size(); ++j) {
        auto store_pid = pid_group.pids(j);
        routing_info[store_pid] = (j % actor_lshard_num) + gsid_offset;
      }
    } else if (worker_type == WorkerType::Serving) {
      if (pid_group.worker_id() == worker_id) {
        for (int j = 0; j < pid_group.pids_size(); ++j) {
          auto store_pid = pid_group.pids(j);
          // serving workers are always standalone(global shard offset = 0)
          routing_info[store_pid] = (j % actor_lshard_num);
        }
      }
    }
  }

  return {partition_strategy, partition_num,
          std::move(managed_pids), std::move(routing_info)};
}

Server::CheckpointInfo RetrieveCheckpointInfo(const CheckpointInfoPb& pb) {
  std::unordered_map<uint32_t, int64_t> ingested_offsets;
  for (auto& tuple: pb.sub_kafka_offsets()) {
    ingested_offsets[tuple.pid()] = tuple.ready_offset();
  }

  std::vector<storage::StorePartitionBackupInfo> sample_store_backup_infos;
  sample_store_backup_infos.reserve(pb.sample_store_backups_size());
  for (auto& tuple: pb.sample_store_backups()) {
    sample_store_backup_infos.emplace_back(
        tuple.pid(), tuple.vertex_bid(), tuple.edge_bid(), tuple.valid());
  }

  std::vector<storage::SubsPartitionBackupInfo> subs_table_backup_infos;
  subs_table_backup_infos.reserve(pb.subs_table_backups_size());
  for (auto& tuple: pb.subs_table_backups()) {
    subs_table_backup_infos.emplace_back(
        tuple.pid(), tuple.bid(), tuple.valid());
  }

  return {std::move(ingested_offsets),
          std::move(sample_store_backup_infos),
          std::move(subs_table_backup_infos)};
}

void Service::RetrieveInitInfo(std::unique_ptr<Server::InitInfo>* info) {
  grpc::ClientContext context;
  GetInitInfoRequestPb  req;
  GetInitInfoResponsePb res;

  req.set_worker_type(worker_type_);
  req.set_worker_id(static_cast<int32_t>(worker_id_));

  auto s = stub_->GetInitInfo(&context, req, &res);
  if (!s.ok()) {
    throw std::runtime_error("Get init info failed: " + s.error_message());
  }

  if (res.terminate_service()) {
    is_termination_.store(true);
    return;
  }

  std::string query_plan;
  uint32_t actor_lshard_num;
  Server::StorePartitionInfo store_partition_info;
  Server::CheckpointInfo checkpoint_info;
  std::unique_ptr<Server::UpstreamInfo> upstream_info;
  std::unique_ptr<Server::DownstreamInfo> downstream_info;

  if (worker_type_ == WorkerType::Sampling) {
    auto& spl_info = res.sampling_info();
    query_plan = spl_info.query_plan();
    actor_lshard_num = spl_info.num_local_shards();
    WriteServerConfigToTmpFile(worker_id_, &res);
    store_partition_info = RetrieveStorePartitionInfo(
        worker_id_, worker_type_, actor_lshard_num,
        spl_info.store_partition_info());
    checkpoint_info = RetrieveCheckpointInfo(spl_info.checkpoint_info());
    upstream_info = RetrieveUpstreamInfo(spl_info.upstream_info());
    downstream_info = RetrieveDownstreamInfo(spl_info.downstream_info());
  } else if (worker_type_ == WorkerType::Serving) {
    auto& srv_info = res.serving_info();
    query_plan = srv_info.query_plan();
    actor_lshard_num = srv_info.num_local_shards();
    store_partition_info = RetrieveStorePartitionInfo(
        worker_id_, worker_type_,
        actor_lshard_num, srv_info.store_partition_info());
    checkpoint_info = RetrieveCheckpointInfo(srv_info.checkpoint_info());
    upstream_info = RetrieveUpstreamInfo(srv_info.upstream_info());
  }

  *info = std::make_unique<Server::InitInfo>(
    worker_type_, worker_id_, num_workers_,
    query_plan, actor_lshard_num,
    std::move(store_partition_info),
    std::move(checkpoint_info),
    std::move(upstream_info),
    std::move(downstream_info));

  LOG(INFO) << "Initialization info is retrieved.";
}

std::unique_ptr<Coordinator::Stub> CreateCoordinatorStub() {
  auto &coord_option = Options::GetInstance().GetCoordClientOptions();
  auto channel = grpc::CreateChannel(coord_option.server_ipaddr,
      grpc::InsecureChannelCredentials());
  auto s = channel->WaitForConnected(gpr_time_add(gpr_now(GPR_CLOCK_REALTIME),
      gpr_time_from_seconds(coord_option.wait_time_in_sec, GPR_TIMESPAN)));

  if (s) {
    LOG(INFO) << "Channel to coordinator is connected";
    return Coordinator::NewStub(channel);
  } else {
    // FIXME(@xmqin): this is a lame channel.
    LOG(FATAL) << "Channel to coordinator can't be established";
  }
  return {nullptr};
}

void WriteServerConfigToTmpFile(uint32_t worker_id,
                                const GetInitInfoResponsePb* res) {
  char list_fname[16];
  snprintf(list_fname, sizeof(list_fname), "s-%d.list", worker_id);
  std::ofstream tmp_file(list_fname);

  auto &spl_info = res->sampling_info();

  tmp_file << "mach_id, #cores, ip_addr" << std::endl;
  for (int32_t i = 0; i < spl_info.ipaddrs_size(); ++i) {
    tmp_file << spl_info.ipaddrs(i) << std::endl;
  }
  tmp_file.close();
}

}  // namespace dgs
