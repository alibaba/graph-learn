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

#ifndef GRAPHLEARN_PLATFORM_HADOOP_HADOOP_FILE_SYSTEM_H_
#define GRAPHLEARN_PLATFORM_HADOOP_HADOOP_FILE_SYSTEM_H_

#include <string>
#include <vector>

#include "graphlearn/common/base/errors.h"
#include "graphlearn/platform/env.h"
#include "graphlearn/platform/file_stats.h"

extern "C" {
  struct hdfs_internal;
  typedef hdfs_internal* hdfsFS;
}

namespace graphlearn {


class LibHDFS;

class HadoopFileSystem : public FileSystem {
public:
  HadoopFileSystem();
  ~HadoopFileSystem();

  Status NewStructuredAccessFile(
      const std::string& path, uint64_t offset, uint64_t end,
      std::unique_ptr<StructuredAccessFile>* result) override;

  Status NewByteStreamAccessFile(
      const std::string& path, uint64_t offset,
      std::unique_ptr<ByteStreamAccessFile>* result) override;

  Status NewWritableFile(
      const std::string& path,
      std::unique_ptr<WritableFile>* result) override {
    return error::Unimplemented("Not implemented yet.");
  }

  Status ListDir(
      const std::string& path,
      std::vector<std::string>* result) override;

  Status GetFileSize(
      const std::string& path,
      uint64_t* size) override {
    return error::Unimplemented("Not implemented yet.");
  }

  Status GetRecordCount(const std::string& path, uint64_t* size) override {
    return error::Unimplemented("HDFS does not support GetRecordCount.");
  }

  Status FileExists(const std::string& path) override {
    return error::Unimplemented("Not implemented yet.");
  }

  Status DeleteFile(const std::string& path) override {
    return error::Unimplemented("Not implemented yet.");
  }

  Status CreateDir(const std::string& path) override {
    return error::Unimplemented("Not implemented yet.");
  }

  Status DeleteDir(const std::string& path) override {
    return error::Unimplemented("Not implemented yet.");
  }

  std::string Translate(const std::string& path) const override;

private:
  Status Connect(std::string fname, hdfsFS* fs);
  Status Stat(const std::string& fname, FileStats* stats);
  LibHDFS* hdfs_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_PLATFORM_HADOOP_HADOOP_FILE_SYSTEM_H_