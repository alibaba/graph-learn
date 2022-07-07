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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "common/base/errors.h"
#include "common/base/macros.h"
#include "common/string/string_tool.h"
#include "core/io/data_slicer.h"
#include "platform/env.h"

namespace graphlearn {
namespace io {

template <class SourceType>
class SliceReader {
public:
  SliceReader(const std::vector<SourceType>& source,
              Env* env,
              int32_t thread_id,
              int32_t thread_num)
      : env_(env), thread_id_(thread_id), thread_num_(thread_num),
        files_cursor_(-1), current_(nullptr),
        offset_(0), end_(0) {
    ReorgSources(source);
  }

  Status BeginNextFile(SourceType** ret) {
    ++files_cursor_;

    if (files_cursor_ >= source_.size()) {
      return error::OutOfRange("All files completed");
    }

    current_ = &source_[files_cursor_];

    FileSystem* fs = NULL;
    Status s = env_->GetFileSystem(current_->path, &fs);
    LOG_RETURN_IF_NOT_OK(s)

    if (! SingleThreadMode(current_->path)) {
      uint64_t file_size = 0;
      s = fs->GetRecordCount(current_->path, &file_size);
      LOG_RETURN_IF_NOT_OK(s)

      int32_t slice_id = 0;
      int32_t slice_count = 1;
      if (IsDistributeShared(current_->path)) {
        slice_id = env_->GetServerId() * thread_num_ + thread_id_;
        slice_count = env_->GetServerCount() * thread_num_;
      } else if (current_->local_shared) {
        slice_id = thread_id_;
        slice_count = thread_num_;
      }

      DataSlicer slicer(slice_id, slice_count, file_size);
      offset_ = slicer.LocalStart();
      end_ = offset_ + slicer.LocalSize();
      LOG(INFO) << "file_size:" << file_size
                << "thread id:" << thread_id_
                << ", thread num:" << thread_num_
                << ", offset:" << offset_
                << ", end:" << end_;
      s = fs->NewStructuredAccessFile(current_->path, offset_, end_, &reader_);
    } else {
      s = fs->NewStructuredAccessFile(current_->path, 0, 0, &reader_);
    }
    RETURN_IF_NOT_OK(s)

    // set schema for hdfsFS
    std::vector<DataType> types;
    types.push_back(kInt64);
    if (std::is_same<typename std::decay<SourceType>::type, EdgeSource>::value){
      types.push_back(kInt64);
    }
    if (current_->IsWeighted()) {
      types.push_back(kFloat);
    }
    if (current_->IsLabeled()) {
      types.push_back(kInt32);
    }
    if (current_->IsAttributed()) {
      types.push_back(kString);
    }
    reader_->SetSchema(types);

    schema_ = reader_->GetSchema();
    *ret = current_;
    return s;
  }

  Status Read(Record* record) {
    if (!SingleThreadMode(current_->path)) {
      if (offset_ >= end_) {
        return error::OutOfRange("Current file completed");
      }
    } else {
      if (thread_id_ != 0) {
        return error::OutOfRange("Just return in Single Thread Mode.");
      }
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
  void ReorgSources(const std::vector<SourceType>& sources) {
    int32_t index = 0;
    for (auto& src : sources) {
      if (IsDistributeShared(src.path)) {
        source_.emplace_back(src);
      } else if (!IsDirectory(src.path)) {
        source_.emplace_back(src);
      } else {
        std::vector<std::string> files;
        ListFiles(src.path, &files);

        for (auto& file : files) {
          if (index++ % env_->GetServerCount() == env_->GetServerId()) {
            source_.emplace_back(src);
            source_.back().path += file;
            source_.back().local_shared = false;
          }
        }
      }
    }
  }

  void ListFiles(const std::string& path, std::vector<std::string>* files) {
    FileSystem* fs = nullptr;
    Status s = env_->GetFileSystem(path, &fs);
    if (!s.ok()) {
      return;
    }
    s = fs->ListDir(path, files);
    if (!s.ok()) {
      LOG(ERROR) << "List directory failed: " << path
                 << ", details: " << s.ToString();
      files->clear();
    }
    std::sort(files->begin(), files->end());
  }

  bool IsDistributeShared(const std::string& path) const {
    return ::graphlearn::strings::StartWith(path, "odps://");
  }

  bool IsDirectory(const std::string& path) const {
    return ::graphlearn::strings::EndWith(path, "/");
  }

  bool SingleThreadMode(const std::string& path) const {
    return ::graphlearn::strings::StartWith(path, "hdfs://") || 
        ::graphlearn::strings::StartWith(path, "viewfs://") ||
        ::graphlearn::strings::StartWith(path, "file://");
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
