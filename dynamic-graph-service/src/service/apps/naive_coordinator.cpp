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

#include <semaphore.h>
#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>

#include "grpcpp/create_channel.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "seastar/core/print.hh"

#include "generated/proto/coordinator.pb.h"
#include "generated/proto/coordinator.grpc.pb.h"

static uint32_t g_num_workers = 1;

class NaiveCoordinatorImpl : public dgs::Coordinator::Service {
public:
  NaiveCoordinatorImpl() : dgs::Coordinator::Service() {
    sem_init(&ready_for_notify_, 0, 0);
    sem_init(&all_is_inited_, 0, 0);
    sampling_worker_ipaddrs_.resize(g_num_workers);
  }

  ~NaiveCoordinatorImpl() override {
    sem_destroy(&ready_for_notify_);
    sem_destroy(&all_is_inited_);
  }

  grpc::Status RegisterWorker(grpc::ServerContext* context,
                              const dgs::RegisterWorkerRequestPb* request,
                              dgs::RegisterWorkerResponsePb* response) override {
    if (request->worker_type() == dgs::WorkerType::Sampling) {
      ip_addrs_mtx_.lock();
      sampling_worker_ipaddrs_[request->worker_id()] = request->worker_ip();
      ip_addrs_mtx_.unlock();
      if (++num_sampling_workers_ == g_num_workers) {
        sem_post(&ready_for_notify_);
      }
    }
    response->set_suc(true);
    response->set_num_workers(g_num_workers);
    fmt::print("Inside RegisterWorker. Worker id is: {}\n", request->worker_id());
    return grpc::Status::OK;
  }

  grpc::Status GetInitInfo(grpc::ServerContext* context,
                           const dgs::GetInitInfoRequestPb* request,
                           dgs::GetInitInfoResponsePb* response) override {
    static const uint32_t num_local_shards = 2;

    std::ifstream f("../../conf/install_query.ut.json");
    std::string query_plan;

    f.seekg(0, std::ios::end);
    query_plan.reserve(f.tellg());
    f.seekg(0, std::ios::beg);

    query_plan.assign((
        std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());

    const uint32_t sampling_store_partition_num = 8;
    const uint32_t serving_store_partition_num  = 6;

    const std::string dl2sampling_kafka_server = "localhost:9092";
    const std::string dl2sampling_kafka_topic = "service-record-polling-ut";
    const uint32_t dl2sampling_kafka_partition_num = 4;

    const std::string sampling2serving_kafka_server = "localhost:9092";
    const std::string sampling2serving_kafka_topic = "service-sample-publishing-ut";
    const uint32_t sampling2serving_kafka_partition_num = 4;

    if (request->worker_type() == dgs::WorkerType::Sampling) {
      sem_wait(&ready_for_notify_);

      dgs::StorePartitionInfoPb store_partition_info;
      store_partition_info.set_partition_strategy("hash");
      store_partition_info.set_partition_num(sampling_store_partition_num);
      for (uint32_t wid = 0; wid < g_num_workers; ++wid) {
        auto managed_pids = store_partition_info.add_managed_pids_group();
        managed_pids->set_worker_id(wid);
        for (uint32_t i = 0; i < sampling_store_partition_num; ++i) {
          if (i % g_num_workers == wid) {
            managed_pids->add_pids(i);
          }
        }
      }

      dgs::UpStreamInfoPb upstream_info;
      upstream_info.add_sub_kafka_servers(dl2sampling_kafka_server);
      upstream_info.set_sub_kafka_topic(dl2sampling_kafka_topic);
      upstream_info.set_sub_kafka_partition_num(dl2sampling_kafka_partition_num);
      for (uint32_t i = 0; i < dl2sampling_kafka_partition_num; ++i) {
        upstream_info.add_sub_kafka_pids(i);
      }

      dgs::DownStreamInfoPb downstream_info;
      downstream_info.set_store_partition_strategy("hash");
      downstream_info.set_store_partition_num(serving_store_partition_num);
      downstream_info.add_pub_kafka_servers(sampling2serving_kafka_server);
      downstream_info.set_pub_kafka_topic(sampling2serving_kafka_topic);
      downstream_info.set_store_partition_num(sampling2serving_kafka_partition_num);
      downstream_info.set_worker_partition_num(1);
      downstream_info.set_worker_partition_strategy("hash");
      for (uint32_t i = 0; i < sampling2serving_kafka_partition_num; ++i) {
        downstream_info.add_pub_kafka_pids(i);
      }

      dgs::CheckpointInfoPb checkpoint_info;
      for (uint32_t i = 0; i < dl2sampling_kafka_partition_num; ++i) {
        auto* entry = checkpoint_info.add_sub_kafka_offsets();
        entry->set_pid(i);
        entry->set_ready_offset(-1);
      }
      for (uint32_t i = 0; i < sampling_store_partition_num; ++i) {
        if (i % g_num_workers == request->worker_id()) {
          auto* store_entry = checkpoint_info.add_sample_store_backups();
          store_entry->set_pid(i);
          store_entry->set_valid(false);
          store_entry->set_vertex_bid(0);
          store_entry->set_edge_bid(0);
          auto* subs_entry = checkpoint_info.add_subs_table_backups();
          subs_entry->set_pid(i);
          subs_entry->set_valid(false);
          subs_entry->set_bid(0);
        }
      }

      dgs::SamplingInitInfoPb info;
      info.set_query_plan(query_plan);
      info.set_num_local_shards(num_local_shards);
      for (uint32_t i = 0; i < sampling_worker_ipaddrs_.size(); ++i) {
        info.add_ipaddrs(std::to_string(i) + " " +
            std::to_string(num_local_shards) + " " +
            sampling_worker_ipaddrs_[i]);
      }
      *info.mutable_upstream_info() = upstream_info;
      *info.mutable_downstream_info() = downstream_info;
      *info.mutable_store_partition_info() = store_partition_info;
      *info.mutable_checkpoint_info() = checkpoint_info;

      *response->mutable_sampling_info() = info;
    } else {
      dgs::StorePartitionInfoPb store_partition_info;
      store_partition_info.set_partition_strategy("hash");
      store_partition_info.set_partition_num(serving_store_partition_num);
      for (uint32_t wid = 0; wid < g_num_workers; ++wid) {
        auto managed_pids = store_partition_info.add_managed_pids_group();
        managed_pids->set_worker_id(wid);
        for (uint32_t i = 0; i < serving_store_partition_num; ++i) {
          if (i % g_num_workers == wid) {
            managed_pids->add_pids(i);
          }
        }
      }

      dgs::UpStreamInfoPb upstream_info;
      upstream_info.add_sub_kafka_servers(sampling2serving_kafka_server);
      upstream_info.set_sub_kafka_topic(sampling2serving_kafka_topic);
      upstream_info.set_sub_kafka_partition_num(sampling2serving_kafka_partition_num);
      for (uint32_t i = 0; i < sampling2serving_kafka_partition_num; ++i) {
        upstream_info.add_sub_kafka_pids(i);
      }

      dgs::CheckpointInfoPb checkpoint_info;
      for (uint32_t i = 0; i < sampling2serving_kafka_partition_num; ++i) {
        auto* entry = checkpoint_info.add_sub_kafka_offsets();
        entry->set_pid(i);
        entry->set_ready_offset(-1);
      }
      for (uint32_t i = 0; i < sampling_store_partition_num; ++i) {
        if (i % g_num_workers == request->worker_id()) {
          auto* store_entry = checkpoint_info.add_sample_store_backups();
          store_entry->set_pid(i);
          store_entry->set_valid(false);
          store_entry->set_vertex_bid(0);
          store_entry->set_edge_bid(0);
        }
      }

      dgs::ServingInitInfoPb info;
      info.set_query_plan(query_plan);
      info.set_num_local_shards(num_local_shards);
      *info.mutable_upstream_info() = upstream_info;
      *info.mutable_store_partition_info() = store_partition_info;
      *info.mutable_checkpoint_info() = checkpoint_info;

      *response->mutable_serving_info() = info;
    }
    response->set_terminate_service(false);

    fmt::print("Inside GetInitInfo, worker id is: {}\n", request->worker_id());

    return grpc::Status::OK;
  }

  grpc::Status ReportInited(grpc::ServerContext* context,
                            const dgs::ReportInitedRequestPb* request,
                            dgs::ReportInitedResponsePb* response) override {
    if (request->worker_type() == dgs::WorkerType::Sampling) {
      if (++num_inited_sampling_workers < g_num_workers) {
        sem_wait(&all_is_inited_);
      } else {
        sem_post(&all_is_inited_);
      }
    }

    fmt::print("Inside ReportInited. Request worker id is: : {}\n", request->worker_id());

    return {};
  }

  grpc::Status GetCheckReadyInfo(grpc::ServerContext* context,
                                 const dgs::CheckReadyInfoRequestPb* request,
                                 dgs::CheckReadyInfoResponsePb* response) override {
    fmt::print("Inside GetCheckReadyInfo. Request worker id is: : {}\n", request->worker_id());

    return {};
  }

  grpc::Status ReportServerIsReady(grpc::ServerContext* context,
                                   const dgs::ServerIsReadyRequestPb* request,
                                   dgs::ServerIsReadyResponsePb* response) override {
    fmt::print("Inside ReportServerIsReady. Request worker id is: : {}\n", request->worker_id());

    return {};
  }

  grpc::Status ReportStatistics(grpc::ServerContext* context,
                                const dgs::ReportStatisticsRequestPb* request,
                                dgs::ReportStatisticsResponsePb* response) override {
    static std::atomic<int> counter{0};
    fmt::print("Inside ReportStatistics. Request worker id is: : {}\n", request->worker_id());

    if (++counter >= 2) {
      response->set_terminate_service(true);
    } else {
      response->set_terminate_service(false);
    }

    return {};
  }

private:
  sem_t            ready_for_notify_;
  sem_t            all_is_inited_;
  std::atomic<int> num_sampling_workers_{0};
  std::atomic<int> num_inited_sampling_workers{0};

  std::vector<std::string> sampling_worker_ipaddrs_;
  std::mutex               ip_addrs_mtx_;
};

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  NaiveCoordinatorImpl coordinator;

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&coordinator);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "#arguments must be equal to 2" << std::endl;
    return -1;
  }

  g_num_workers = strtol(argv[1], nullptr, 10);
  std::cout << "Number of workers = " << g_num_workers << std::endl;

  RunServer();
}
