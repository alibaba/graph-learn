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

#include "dataloader/service.h"

#include <fstream>
#include <thread>

#include "grpc/support/time.h"
#include "grpcpp/create_channel.h"

#include "dataloader/host.h"
#include "dataloader/logging.h"

namespace dgs {
namespace dataloader {

Service::Service(const std::string& config_file, int32_t worker_id): worker_id_(worker_id) {
  InitGoogleLogging();
  Configure(config_file);
  CreateStub();
}

Service::~Service() {
  UninitGoogleLogging();
}

void Service::Run() {
  RegisterSelf();

  std::thread reporter;

  GetInfoAndInit();
  if (is_termination_.load(std::memory_order_relaxed)) {
    goto FINAL;
  }

  WaitAllInited();
  if (is_termination_.load(std::memory_order_relaxed)) {
    goto FINAL;
  }

  reporter = std::thread(&Service::ReportStatsInfo, this);

  BulkLoad();

  ReportReady();
  if (is_termination_.load(std::memory_order_relaxed)) {
    goto FINAL;
  }

  StreamingLoad();

FINAL:
  if (reporter.joinable()) {
    reporter.join();
  }
  LOG(INFO) << "Data loading service is terminated...";
}

void Service::Configure(const std::string& config_file) const {
  if (!Options::GetInstance().LoadFile(config_file)) {
    throw std::runtime_error("Configuring data loading service option failed");
  }
  Options::GetInstance().loader_id = worker_id_;
  if (!Schema::GetInstance().Init()) {
    throw std::runtime_error("Configuring graph schema failed");
  }
  LOG(INFO) << "Configure data loading service successful with config file: " << config_file;
}

void Service::CreateStub() {
  auto &opts = Options::GetInstance();
  auto channel = grpc::CreateChannel(opts.coord_ipaddr, grpc::InsecureChannelCredentials());
  auto ok = channel->WaitForConnected(
      gpr_time_add(gpr_now(GPR_CLOCK_REALTIME), gpr_time_from_seconds(opts.coord_connect_timeout_sec, GPR_TIMESPAN)));
  if (!ok) {
    throw std::runtime_error("Channel to coordinator from dataloader can't be established!");
  }
  stub_ = Coordinator::NewStub(channel);
  LOG(INFO) << "Channel to coordinator from dataloader is connected";
}

void Service::RegisterSelf() {
  grpc::ClientContext context;
  RegisterWorkerRequestPb req;
  RegisterWorkerResponsePb res;

  auto ipaddr = GetLocalEndpoint(GetAvailablePort());
  req.set_worker_type(worker_type_);
  req.set_worker_id(worker_id_);
  req.set_worker_ip(ipaddr);

  auto s = stub_->RegisterWorker(&context, req, &res);
  if (!s.ok()) {
    throw std::runtime_error("Failed to register current dataloader to coordinator: " + s.error_message());
  }
  if (!res.suc()) {
    throw std::runtime_error("Failed to register current dataloader with worker id: " + std::to_string(worker_id_));
  }

  auto& opts = Options::GetInstance();
  opts.loader_num = num_workers_ = res.num_workers();
  LOG(INFO) << "Registered current dataloader with worker id: " << worker_id_ << ", total dataloader number: " << num_workers_;
}

void Service::GetInfoAndInit() {
  grpc::ClientContext context;
  GetInitInfoRequestPb req;
  GetInitInfoResponsePb res;

  req.set_worker_type(worker_type_);
  req.set_worker_id(worker_id_);

  auto s = stub_->GetInitInfo(&context, req, &res);
  if (!s.ok()) {
    throw std::runtime_error("Get init info failed: " + s.error_message());
  }
  if (res.terminate_service()) {
    is_termination_.store(true);
    return;
  }

  auto& opts = Options::GetInstance();
  auto& ds_kafka_info = res.dataloader_info().ds_kafka_info();
  auto& ds_partition_info = res.dataloader_info().ds_store_partition_info();
  opts.data_partitions = ds_partition_info.store_partition_num();
  opts.output_kafka_brokers.resize(ds_kafka_info.pub_kafka_servers_size());
  for (int i = 0; i < ds_kafka_info.pub_kafka_servers_size(); i++) {
    opts.output_kafka_brokers[i] = ds_kafka_info.pub_kafka_servers(i);
  }
  opts.output_kafka_topic = ds_kafka_info.pub_kafka_topic();
  opts.output_kafka_partitions = ds_kafka_info.pub_kafka_partition_num();
  std::vector<PartitionId> kafka_router;
  for (auto pid : ds_partition_info.store_to_kafka_pid_vec()) {
    kafka_router.push_back(pid);
  }

  LOG(INFO) << "-- output data partitions: " << opts.data_partitions;
  LOG(INFO) << "-- output kafka brokers: " << opts.FormatOutputKafkaBrokers();
  LOG(INFO) << "-- output kafka topic: " << opts.output_kafka_topic;
  LOG(INFO) << "-- output kafka partitions: " << opts.output_kafka_partitions;
  std::string router_info;
  for (auto pid : kafka_router) {
    router_info += std::to_string(pid) + " ";
  }
  LOG(INFO) << "-- managed kafka partition ids: " << router_info;

  Partitioner::GetInstance().Set(ds_partition_info.store_partition_strategy(), std::move(kafka_router));

  LOG(INFO) << "Data loading service is initialized.";
}

void Service::WaitAllInited() {
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

void Service::ReportReady() {
  ServerIsReadyRequestPb req;
  ServerIsReadyResponsePb res;
  grpc::ClientContext ctx;

  req.set_worker_type(worker_type_);
  req.set_worker_id(worker_id_);

  auto s = stub_->ReportServerIsReady(&ctx, req, &res);
  if (!s.ok()) {
    throw std::runtime_error(s.error_message());
  }

  if (res.terminate_service()) {
    is_termination_.store(true);
  }

  LOG(INFO) << "Data loading service is ready.";
}

void Service::ReportStatsInfo() {
  ReportStatisticsRequestPb req;
  ReportStatisticsResponsePb res;
  grpc::Status s;

  req.set_worker_id(worker_id_);
  req.set_worker_type(worker_type_);

  auto heartbeat_interval = Options::GetInstance().coord_heartbeat_interval_sec;
  uint32_t failed_times = 0;

  while (!is_termination_) {
    // report statistics with a fixed interval
    // the first report will begin after an initial delay.
    std::this_thread::sleep_for(std::chrono::seconds(heartbeat_interval));

    grpc::ClientContext context;
    s = stub_->ReportStatistics(&context, req, &res);
    if (!s.ok()) {
      failed_times++;
      LOG(WARNING) << "Failed to report statistics by " << failed_times << " times";
      if (failed_times >= 3) {
        LOG(WARNING) << "Start to terminate data loading service ...";
        is_termination_.store(true);
        break;
      }
    } else {
      failed_times = 0;
      if (res.terminate_service()) {
        is_termination_.store(true);
        break;
      }
    }
  }

  WriteTerminateFlag();
}

void Service::WriteTerminateFlag() {
  std::ofstream file("./TERMINATE_FLAG");
  file << "TERMINATED";
  file.close();
}

}  // namespace dataloader
}  // namespace dgs
