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

#ifndef FILE_LOADER_LOADER_H_
#define FILE_LOADER_LOADER_H_

#include "dataloader/service.h"

#include "line_processer.h"

namespace dgs {
namespace dataloader {
namespace file {

class FileLoader {
public:
  FileLoader() = default;
  ~FileLoader() = default;

  void Load(const std::string& file_path);

private:
  GroupProducer group_producer_;
};

class FileLoadingService : public Service {
public:
  FileLoadingService(const std::string& config_file, int32_t worker_id, const std::string& pattern_def_file,
                     std::string&& bulk_load_file, std::string&& streaming_load_file)
    : Service(config_file, worker_id),
      bulk_load_file_(std::move(bulk_load_file)),
      streaming_load_file_(std::move(streaming_load_file)) {
    LineProcessor::GetInstance().Init(pattern_def_file);
  }
  ~FileLoadingService() = default;

protected:
  void BulkLoad() override {
    if (!bulk_load_file_.empty()) {
      FileLoader loader;
      loader.Load(bulk_load_file_);
      LOG(INFO) << "[Bulk Load] Finish loading file: " << bulk_load_file_;
    }
  }

  void StreamingLoad() override {
    if (!streaming_load_file_.empty()) {
      FileLoader loader;
      loader.Load(streaming_load_file_);
      LOG(INFO) << "[Streaming Load] Finish loading file: " << streaming_load_file_;
    }
  }

private:
  const std::string bulk_load_file_;
  const std::string streaming_load_file_;
};

}  // namespace file
}  // namespace dataloader
}  // namespace dgs

#endif // FILE_LOADER_LOADER_H_
