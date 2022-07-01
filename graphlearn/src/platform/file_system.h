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

#ifndef GRAPHLEARN_PLATFORM_FILE_SYSTEM_H_
#define GRAPHLEARN_PLATFORM_FILE_SYSTEM_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "graphlearn/common/base/uncopyable.h"
#include "graphlearn/common/io/value.h"
#include "graphlearn/common/string/lite_string.h"
#include "graphlearn/include/status.h"

namespace graphlearn {

class ByteStreamAccessFile;
class StructuredAccessFile;
class WritableFile;

class FileSystem {
public:
  FileSystem() {}

  virtual ~FileSystem() = default;

  virtual Status NewByteStreamAccessFile(
      const std::string& file_name, uint64_t offset,
      std::unique_ptr<ByteStreamAccessFile>* f) = 0;

  virtual Status NewStructuredAccessFile(
      const std::string& file_name, uint64_t offset, uint64_t end,
      std::unique_ptr<StructuredAccessFile>* f) = 0;

  virtual Status NewWritableFile(
      const std::string& file_name,
      std::unique_ptr<WritableFile>* f) = 0;

  virtual Status ListDir(
      const std::string& dir_name,
      std::vector<std::string>* result) = 0;

  virtual Status GetFileSize(
      const std::string& file_name,
      uint64_t* file_size) = 0;

  virtual Status GetRecordCount(
      const std::string& file_name,
      uint64_t* count) = 0;

  virtual Status FileExists(const std::string& file_name) = 0;
  virtual Status DeleteFile(const std::string& file_name) = 0;
  virtual Status CreateDir(const std::string& dir_name) = 0;
  virtual Status DeleteDir(const std::string& dir_name) = 0;

  virtual std::string Translate(const std::string& name) const {
    return name;
  }
};

class ByteStreamAccessFile : private Uncopyable {
public:
  explicit ByteStreamAccessFile(uint64_t offset) : offset_(offset) {}
  virtual ~ByteStreamAccessFile() = default;

  /// Reads next `n` bytes.
  /// Data will be stored in `buffer`, and `result` refers to `buffer`.
  /// Be sure that, the size of buffer must be >= n.
  /// If n bytes exist, return OK.
  /// If fewer than n bytes exist and not reach the end, return OK.
  /// If no bytes exist, return OUT_OF_RANGE.
  virtual Status Read(size_t n, LiteString* result, char* buffer) = 0;

protected:
  uint64_t offset_;
};

class StructuredAccessFile : private Uncopyable {
public:
  StructuredAccessFile(uint64_t offset, uint64_t end = -1)
    : offset_(offset), end_(end) {}
  virtual ~StructuredAccessFile() = default;

  /// Reads next data record into `result`.
  /// If no records exist, return OUT_OF_RANGE.
  virtual Status Read(io::Record* result) = 0;

  virtual const io::Schema& GetSchema() const = 0;

  virtual void SetSchema(const std::vector<DataType>& types) {
    schema_ = io::Schema(types);
  }

protected:
  uint64_t offset_;
  uint64_t end_;
  io::Schema schema_;
};

class WritableFile : private Uncopyable {
public:
  WritableFile() {}
  virtual ~WritableFile() = default;

  /// Append `data` to the end of the file.
  virtual Status Append(const LiteString& data) = 0;

  virtual Status Flush() = 0;
  virtual Status Close() = 0;
};

class FileSystemRegistry {
public:
  typedef std::function<FileSystem*()> Factory;

  virtual ~FileSystemRegistry() = default;
  virtual Status Register(const std::string& scheme, Factory factory) = 0;
  virtual FileSystem* Lookup(const std::string& scheme) = 0;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_PLATFORM_FILE_SYSTEM_H_
