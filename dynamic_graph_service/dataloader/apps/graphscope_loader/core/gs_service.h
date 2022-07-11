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

#ifndef GRAPHSCOPE_LOADER_GS_SERVICE_H_
#define GRAPHSCOPE_LOADER_GS_SERVICE_H_

#include "dataloader/service.h"

#include "bulk_loader.h"
#include "log_poller.h"

namespace dgs {
namespace dataloader {
namespace gs {

class GraphscopeLoadingService : public Service {
public:
  GraphscopeLoadingService(const std::string& config_file, int32_t worker_id);
  ~GraphscopeLoadingService();

protected:
  /// Run graphscope-store initial bulk loading tasks.
  void BulkLoad() override;

  /// Run graphscope-store log polling tasks.
  void StreamingLoad() override;

private:
  static void WriteBulkLoadFinishFlag();
  static bool CheckBulkLoadFinishFlag();
};

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs

#endif // GRAPHSCOPE_LOADER_GS_SERVICE_H_
