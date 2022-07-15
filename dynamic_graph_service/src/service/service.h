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

#ifndef DGS_SERVICE_SERVICE_H_
#define DGS_SERVICE_SERVICE_H_

#include "boost/asio.hpp"

#include "common/options.h"
#include "generated/proto/coordinator.grpc.pb.h"
#include "service/server.h"

namespace dgs {

class Service {
public:
  explicit Service(const std::string& config_file, uint32_t worker_id);
  ~Service();

  void Run();

private:
  void Configure(const std::string& config_file);
  void RegisterSelf();
  void RetrieveInitInfo(std::unique_ptr<Server::InitInfo>* info);
  void ReportSelfIsStarted();
  void ReportStatsInfo();
  static void WriteTerminateFlag();

private:
  WorkerType worker_type_;
  uint32_t   worker_id_;
  uint32_t   num_workers_;
  std::atomic<bool>                  is_termination_;
  std::unique_ptr<Server>            server_;
  std::unique_ptr<Coordinator::Stub> stub_;
  boost::asio::thread_pool           backup_executor_{1};
};

}  // namespace dgs

#endif  // DGS_SERVICE_SERVICE_H_
