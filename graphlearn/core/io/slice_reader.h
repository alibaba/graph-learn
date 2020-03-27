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

#ifndef GRAPHLEARN_CORE_IO_SLICE_READER_H_
#define GRAPHLEARN_CORE_IO_SLICE_READER_H_

#include <cstdint>
#include <memory>
#include <vector>
#include "graphlearn/common/base/macros.h"
#include "graphlearn/core/io/data_slicer.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {
namespace io {

template <class SourceType>
class SliceReader {
public:
  SliceReader(const std::vector<SourceType>& source,
              Env* env,
              int32_t thread_id,
              int32_t thread_num)
      : source_(source), env_(env),
        thread_id_(thread_id), thread_num_(thread_num),
        files_cursor_(-1), current_(nullptr),
        offset_(0), end_(0) {
  }

  Status BeginNextFile(SourceType** ret) {
    ++files_cursor_;

    if (files_cursor_ >= source_.size()) {
      return error::OutOfRange("All files completed");
    }

    current_ = &source_[files_cursor_];

    FileSystem* fs = NULL;
    Status s = env_->GetFileSystem(current_->path, &fs);
    RETURN_IF_NOT_OK(s)

    uint64_t file_size = 0;
    s = fs->GetRecordCount(current_->path, &file_size);
    RETURN_IF_NOT_OK(s)

    DataSlicer slicer(env_->GetServerId() * thread_num_ + thread_id_,
                      env_->GetServerCount() * thread_num_,
                      file_size);
    offset_ = slicer.LocalStart();
    end_ = offset_ + slicer.LocalSize();

    s = fs->NewStructuredAccessFile(current_->path, offset_, &reader_);
    RETURN_IF_NOT_OK(s)

    schema_ = reader_->GetSchema();
    *ret = current_;
    return s;
  }

  Status Read(Record* record) {
    if (offset_ >= end_) {
      return error::OutOfRange("Current file completed");
    }

    Status s = reader_->Read(record);
    RETURN_IF_NOT_OK(s)

    ++offset_;
    return s;
  }

  const Schema* GetSchema() {
    return &schema_;
  }

private:
  Env*    env_;
  int32_t thread_id_;
  int32_t thread_num_;
  int32_t files_cursor_;
  int64_t offset_;
  int64_t end_;

  std::vector<SourceType> source_;
  SourceType*             current_;
  Schema                  schema_;
  std::unique_ptr<StructuredAccessFile> reader_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_IO_SLICE_READER_H_
