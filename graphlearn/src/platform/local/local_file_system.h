/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_PLATFORM_LOCAL_LOCAL_FILE_SYSTEM_H_
#define GRAPHLEARN_PLATFORM_LOCAL_LOCAL_FILE_SYSTEM_H_

#include <memory>
#include <string>
#include <vector>
#include "graphlearn/platform/env.h"

namespace graphlearn {

class LocalFileSystem : public FileSystem {
public:
  LocalFileSystem() {}
  ~LocalFileSystem() = default;

  Status NewByteStreamAccessFile(
      const std::string& path, uint64_t offset,
      std::unique_ptr<ByteStreamAccessFile>* result) override;

  Status NewStructuredAccessFile(
      const std::string& path, uint64_t offset, uint64_t end,
      std::unique_ptr<StructuredAccessFile>* result) override;

  Status NewWritableFile(
      const std::string& path,
      std::unique_ptr<WritableFile>* result) override;

  Status ListDir(
      const std::string& path,
      std::vector<std::string>* result) override;

  Status GetFileSize(
      const std::string& path,
      uint64_t* size) override;

  Status GetRecordCount(
      const std::string& file_name,
      uint64_t* count) override;

  Status FileExists(const std::string& path) override;
  Status DeleteFile(const std::string& path) override;
  Status CreateDir(const std::string& path) override;
  Status DeleteDir(const std::string& path) override;
  std::string Translate(const std::string& path) const override;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_PLATFORM_LOCAL_LOCAL_FILE_SYSTEM_H_
