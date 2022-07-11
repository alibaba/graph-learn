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

#ifndef DATALOADER_SERVICE_H_
#define DATALOADER_SERVICE_H_

#include "dataloader/batch_builder.h"
#include "dataloader/batch_producer.h"
#include "dataloader/options.h"
#include "dataloader/partitioner.h"
#include "dataloader/schema.h"
#include "dataloader/proto/coordinator.grpc.pb.h"

namespace dgs {
namespace dataloader {

class Service {
public:
  Service(const std::string& config_file, int32_t worker_id);
  ~Service();

  void Run();

protected:
  /// Run initial bulk loading tasks.
  virtual void BulkLoad() {};

  /// Run streaming data loading tasks.
  /// The impl of this func must be non-blocking, running on background threads is recommended.
  virtual void StreamingLoad() {};

private:
  void Configure(const std::string& config_file) const;
  void CreateStub();
  void RegisterSelf();
  void GetInfoAndInit();
  void WaitAllInited();
  void ReportReady();
  void ReportStatsInfo();
  static void WriteTerminateFlag();

private:
  const WorkerType worker_type_ = DataLoader;
  uint32_t worker_id_ = 0;
  uint32_t num_workers_ = 0;
  std::atomic<bool> is_termination_{false};
  std::unique_ptr<Coordinator::Stub> stub_;
};

}  // namespace dataloader
}  // namespace dgs

#endif // DATALOADER_SERVICE_H_
