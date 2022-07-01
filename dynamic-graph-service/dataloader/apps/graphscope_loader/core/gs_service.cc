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

#include "gs_service.h"

#include "boost/filesystem.hpp"

#include "gs_option.h"

namespace fs = boost::filesystem;

namespace dgs {
namespace dataloader {
namespace gs {

GraphscopeLoadingService::GraphscopeLoadingService(const std::string& config_file, int32_t worker_id)
  : Service(config_file, worker_id) {
  bool ok = GSOptions::GetInstance().LoadFile(config_file);
  if (!ok) {
    throw std::runtime_error("Configuring graphscope loading options failed");
  }
}

GraphscopeLoadingService::~GraphscopeLoadingService() {
  LogPollingManager::GetInstance().Finalize();
}

void GraphscopeLoadingService::BulkLoad() {
  if (CheckBulkLoadFinishFlag()) {
    LOG(INFO) << "Bulk loading has already finished, skip.";
    return;
  }
  auto infos = GetBulkLoadingInfos(GSOptions::GetInstance().checkpoint_restore_dir);
  auto& bulk_loading_pool = BulkLoadingThreadPool::GetInstance();
  bulk_loading_pool.Init();
  for (auto& info : infos) {
    bulk_loading_pool.Load(info, [db = info.db_name] {
      LOG(INFO) << "Bulk loading finished with db: " << db;
    });
  }
  bulk_loading_pool.Join();
  bulk_loading_pool.Finalize();
  WriteBulkLoadFinishFlag();
}

void GraphscopeLoadingService::StreamingLoad() {
  auto& polling_manager = LogPollingManager::GetInstance();
  polling_manager.Init(GetLogOffsets(GSOptions::GetInstance().checkpoint_restore_dir));
  polling_manager.Start();
}

void GraphscopeLoadingService::WriteBulkLoadFinishFlag() {
  auto flag_path = GSOptions::GetInstance().bulk_load_meta_dir + "/FINISH_FLAG";
  std::ofstream file(flag_path);
  file << "FINISHED";
  file.close();
}

bool GraphscopeLoadingService::CheckBulkLoadFinishFlag() {
  fs::path flag_path(GSOptions::GetInstance().bulk_load_meta_dir + "/FINISH_FLAG");
  return fs::exists(flag_path);
}

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs
