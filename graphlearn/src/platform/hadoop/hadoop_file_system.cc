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

#include "platform/hadoop/hadoop_file_system.h"

#include <dlfcn.h>
#include <mutex>
#include <stdlib.h>

#include "common/base/errors.h"
#include "common/io/line_iterator.h"
#include "common/io/path_util.h"
#include "common/io/value.h"
#include "common/string/lite_string.h"
#include "common/string/numeric.h"
#include "common/string/string_tool.h"
#include "common/threading/sync/lock.h"
#include "hadoop/hdfs.h"
#include "include/config.h"


namespace graphlearn {


Status LoadDynamicLibrary(const char* library_filename, void** handle) {
  *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);
  if (!*handle) {
    const char* error_msg = dlerror();
    return error::NotFound(error_msg ? error_msg : "(null error message)");
  }
  return Status::OK();
}

Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                            void** symbol) {
  if (!handle) {
    *symbol = nullptr;
  } else {
    *symbol = dlsym(handle, symbol_name);
  }
  if (!*symbol) {
    const char* error_msg = dlerror();
    return error::NotFound(error_msg ? error_msg : "(null error message)");
  }
  return Status::OK();
}

template <typename R, typename... Args>
Status BindFunc(void* handle, const char* name,
                std::function<R(Args...)>* func) {
  void* symbol_ptr = nullptr;
  RETURN_IF_ERROR(GetSymbolFromLibrary(handle, name, &symbol_ptr));
  *func = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
  return Status::OK();
}

class LibHDFS {
public:
  static LibHDFS* Load() {
    static LibHDFS* lib = []() -> LibHDFS* {
      LibHDFS* lib = new LibHDFS;
      lib->LoadAndBind();
      return lib;
    }();

    return lib;
  }

  // The status, if any, from failure to load.
  Status status() { return status_; }

  std::function<hdfsFS(hdfsBuilder*)> hdfsBuilderConnect;
  std::function<hdfsBuilder*()> hdfsNewBuilder;
  std::function<void(hdfsBuilder*, const char*)> hdfsBuilderSetNameNode;
  std::function<int(const char*, char**)> hdfsConfGetStr;
  std::function<void(hdfsBuilder*, const char* kerbTicketCachePath)>
      hdfsBuilderSetKerbTicketCachePath;
  std::function<int(hdfsFS, hdfsFile)> hdfsCloseFile;
  std::function<tSize(hdfsFS, hdfsFile, tOffset, void*, tSize)> hdfsPread;
  std::function<hdfsFile(hdfsFS, const char*, int, int, short, tSize)>
      hdfsOpenFile;
  std::function<hdfsFileInfo*(hdfsFS, const char*, int*)> hdfsListDirectory;
  std::function<void(hdfsFileInfo*, int)> hdfsFreeFileInfo;
  std::function<hdfsFileInfo*(hdfsFS, const char*)> hdfsGetPathInfo;

private:
  void LoadAndBind() {
    auto TryLoadAndBind = [this](const char* name, void** handle) -> Status {
      RETURN_IF_ERROR(LoadDynamicLibrary(name, handle));
#define BIND_HDFS_FUNC(function) \
  RETURN_IF_ERROR(BindFunc(*handle, #function, &function));

      BIND_HDFS_FUNC(hdfsBuilderConnect);
      BIND_HDFS_FUNC(hdfsNewBuilder);
      BIND_HDFS_FUNC(hdfsBuilderSetNameNode);
      BIND_HDFS_FUNC(hdfsConfGetStr);
      BIND_HDFS_FUNC(hdfsBuilderSetKerbTicketCachePath);
      BIND_HDFS_FUNC(hdfsCloseFile);
      BIND_HDFS_FUNC(hdfsPread);
      BIND_HDFS_FUNC(hdfsOpenFile);
      BIND_HDFS_FUNC(hdfsListDirectory);
      BIND_HDFS_FUNC(hdfsFreeFileInfo);
      BIND_HDFS_FUNC(hdfsGetPathInfo);
#undef BIND_HDFS_FUNC
      return Status::OK();
    };

#if defined(PLATFORM_WINDOWS)
    const char* kLibHdfsDso = "hdfs.dll";
#elif defined(MACOS) || defined(TARGET_OS_MAC)
    const char* kLibHdfsDso = "libhdfs.dylib";
#else
    const char* kLibHdfsDso = "libhdfs.so";
#endif
    char* hdfs_home = getenv("HADOOP_HOME");
    if (hdfs_home != nullptr) {
      std::string path = std::string(hdfs_home) + "lib/native/" + kLibHdfsDso;
      status_ = TryLoadAndBind(path.c_str(), &handle_);
      if (status_.ok()) {
        return;
      }
    }
    // Try to load the library dynamically in case it has been installed
    // to a in non-standard location.
    status_ = TryLoadAndBind(kLibHdfsDso, &handle_);
  }

  Status status_;
  void* handle_ = nullptr;
};


class HDFSByteStreamAccessFile : public ByteStreamAccessFile {
public:
  HDFSByteStreamAccessFile(const std::string& filename, 
                           const std::string& hdfs_filename, LibHDFS* hdfs, 
                           hdfsFS fs, hdfsFile file, uint64_t offset)
      : ByteStreamAccessFile(offset),
        filename_(filename),
        hdfs_filename_(hdfs_filename),
        hdfs_(hdfs),
        fs_(fs),
        file_(file) {}

  ~HDFSByteStreamAccessFile() override {
    if (file_ != nullptr) {
      ScopedLocker<std::mutex> _(&mu_);
      hdfs_->hdfsCloseFile(fs_, file_);
    }
  }

  Status Read(size_t n, LiteString* result, char* scratch) override {
    Status s;
    char* dst = scratch;
    bool eof_retried = false;
    while (n > 0 && s.ok()) {
      ScopedLocker<std::mutex> _(&mu_);
      tSize r = hdfs_->hdfsPread(fs_, file_, static_cast<tOffset>(offset_), dst,
                                 static_cast<tSize>(n));
      if (r > 0) {
        dst += r;
        n -= r;
        offset_ += r;
      } else if (!eof_retried && r == 0) {
        if (file_ != nullptr && hdfs_->hdfsCloseFile(fs_, file_) != 0) {
          return error::Internal("Read hdfs file failed: " + filename_);
        }
        file_ =
            hdfs_->hdfsOpenFile(fs_, hdfs_filename_.c_str(), O_RDONLY, 0, 0, 0);
        if (file_ == nullptr) {
          return error::Internal("Read hdfs file failed: " + filename_);
        }
        eof_retried = true;
      } else if (eof_retried && r == 0) {
        s = error::OutOfRange("Read less bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // hdfsPread may return EINTR too. Just retry.
      } else {
        s = error::Internal("Read hdfs file failed: " + filename_);
      }
    }
    *result = LiteString(scratch, dst - scratch);
    return s;
  }

 private:
  std::string filename_;
  std::string hdfs_filename_;
  LibHDFS* hdfs_;
  hdfsFS fs_;

  mutable std::mutex mu_;
  mutable hdfsFile file_;
};


class HadoopStructuredAccessFile : public StructuredAccessFile {
public:
  HadoopStructuredAccessFile(const std::string& filename, 
                             const std::string& hdfs_filename, LibHDFS* hdfs, 
                             hdfsFS fs, hdfsFile file, uint64_t offset)
    : StructuredAccessFile(offset),
      file_(nullptr),
      line_(nullptr) {
    file_ = new HDFSByteStreamAccessFile(
        filename, hdfs_filename, hdfs, fs, file, offset);
    line_ = new io::LineIterator(file_, 2 * 1024 * 1024);
  }

  ~HadoopStructuredAccessFile() override {
    delete file_;
    delete line_;
  }

  Status Read(io::Record* result) override {
    std::string tmp;
    Status s = line_->Next(&tmp);
    if (s.ok()) {
      ParseRecord(tmp, result);
    }
    return s;
  }

  const io::Schema& GetSchema() const override {
    return schema_;
  }

private:
  void ParseRecord(const std::string& line, io::Record* result) const {
    std::string field_delimiter = GLOBAL_FLAG(FieldDelimiter);
    std::vector<std::string> items = strings::Split(line, field_delimiter);
    if (items.size() != schema_.Size()) {
      return;
    }
    for (size_t i = 0; i < items.size(); ++i) {
      io::Value& v = (*result)[i];
      if (schema_.types[i] == DataType::kInt32) {
        strings::FastStringTo32(items[i].c_str(), &v.n.i);
      } else if (schema_.types[i] == DataType::kInt64) {
        strings::FastStringTo64(items[i].c_str(), &v.n.l);
      } else if (schema_.types[i] == DataType::kFloat ||
          schema_.types[i] == DataType::kDouble) {
        strings::FastStringToFloat(items[i].c_str(), &v.n.f);
      } else {
        v.s.Copy(items[i], true);
      }
    }
  }

private:
  ByteStreamAccessFile* file_;
  io::LineIterator*     line_;
};

HadoopFileSystem::HadoopFileSystem() : hdfs_(LibHDFS::Load()) {}

HadoopFileSystem::~HadoopFileSystem() {}

Status HadoopFileSystem::NewByteStreamAccessFile(
    const std::string& fname, uint64_t offset,
    std::unique_ptr<ByteStreamAccessFile>* result) {
  hdfsFS fs = nullptr;
  RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFile file =
      hdfs_->hdfsOpenFile(fs, Translate(fname).c_str(), O_RDONLY, 0, 0, 0);
  if (file == nullptr) {    
    return error::InvalidArgument("Open hdfs file failed");
  }
  result->reset(new HDFSByteStreamAccessFile(fname, Translate(fname), 
                                             hdfs_, fs, file, offset));
  return Status::OK();
}

Status HadoopFileSystem::NewStructuredAccessFile(
    const std::string& fname, uint64_t offset, uint64_t end,
    std::unique_ptr<StructuredAccessFile>* result) {
  hdfsFS fs = nullptr;
  RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFile file =
      hdfs_->hdfsOpenFile(fs, Translate(fname).c_str(), O_RDONLY, 0, 0, 0);
  if (file == nullptr) {    
    return error::InvalidArgument("Open hdfs file failed");
  }
  result->reset(new HadoopStructuredAccessFile(fname, Translate(fname),
                                               hdfs_, fs, file, offset));
  return Status::OK();
}

Status HadoopFileSystem::Connect(std::string fname, hdfsFS* fs) {
  RETURN_IF_ERROR(hdfs_->status());

  std::string scheme, namenode, path;
  io::ParseURI(fname, &scheme, &namenode, &path);

  hdfsBuilder* builder = hdfs_->hdfsNewBuilder();
  if (scheme == "file") {
    hdfs_->hdfsBuilderSetNameNode(builder, nullptr);
  } else if (scheme == "viewfs") {
    char* defaultFS = nullptr;
    hdfs_->hdfsConfGetStr("fs.defaultFS", &defaultFS);
    std::string defaultScheme, defaultCluster, defaultPath;
    io::ParseURI(defaultFS, &defaultScheme, &defaultCluster, &defaultPath);

    if (scheme != defaultScheme || namenode != defaultCluster) {
      return error::Unimplemented(
          "viewfs is only supported as a fs.defaultFS.");
    }
    // The default NameNode configuration will be used (from the XML
    // configuration files).
    hdfs_->hdfsBuilderSetNameNode(builder, "default");
  } else {
    hdfs_->hdfsBuilderSetNameNode(builder, namenode.c_str());
  }
  char* ticket_cache_path = getenv("KERB_TICKET_CACHE_PATH");
  if (ticket_cache_path != nullptr) {
    hdfs_->hdfsBuilderSetKerbTicketCachePath(builder, ticket_cache_path);
  }
  *fs = hdfs_->hdfsBuilderConnect(builder);
  if (*fs == nullptr) {
    return error::NotFound(fname + " not found");
  }
  return Status::OK();
}

std::string HadoopFileSystem::Translate(const std::string& name) const {
  std::string scheme, namenode, path;
  io::ParseURI(name, &scheme, &namenode, &path);
  return path;
}

Status HadoopFileSystem::ListDir(const std::string& dir,
                                 std::vector<std::string>* result) {
  result->clear();
  hdfsFS fs = nullptr;
  RETURN_IF_ERROR(Connect(dir, &fs));

  // hdfsListDirectory returns nullptr if the directory is empty. Do a separate
  // check to verify the directory exists first.
  FileStats stat;
  RETURN_IF_ERROR(Stat(dir, &stat));

  int entries = 0;
  hdfsFileInfo* info =
      hdfs_->hdfsListDirectory(fs, Translate(dir).c_str(), &entries);
  if (info == nullptr) {
    if (stat.is_directory) {
      // Assume it's an empty directory.
      return Status::OK();
    }
    return error::Internal("List hdfs dir failed: " + dir);
  }
  for (int i = 0; i < entries; i++) {
    result->push_back(std::string(io::BaseName(info[i].mName)));
  }
  hdfs_->hdfsFreeFileInfo(info, entries);
  return Status::OK();
}

Status HadoopFileSystem::Stat(const std::string& fname, FileStats* stats) {
  hdfsFS fs = nullptr;
  RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFileInfo* info = hdfs_->hdfsGetPathInfo(fs, Translate(fname).c_str());
  if (info == nullptr) {
    return error::InvalidArgument("Get hdfs path failed");
  }
  stats->length = static_cast<int64_t>(info->mSize);
  stats->mtime_nsec = static_cast<int64_t>(info->mLastMod) * 1e9;
  stats->is_directory = info->mKind == kObjectKindDirectory;
  hdfs_->hdfsFreeFileInfo(info, 1);
  return Status::OK();
}

REGISTER_FILE_SYSTEM("hdfs", HadoopFileSystem);
REGISTER_FILE_SYSTEM("viewfs", HadoopFileSystem);
REGISTER_FILE_SYSTEM("file", HadoopFileSystem);

}  // namespace graphlearn