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

#include "graphlearn/platform/env.h"

#include <mutex>  //NOLINT [build/c++11]
#include <unordered_map>
#include <utility>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/io/path_util.h"
#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/include/config.h"

namespace graphlearn {

class FileSystemRegistryImpl : public FileSystemRegistry {
public:
  Status Register(const std::string& scheme, Factory factory) override;
  FileSystem* Lookup(const std::string& scheme) override;

private:
  std::mutex mu_;
  std::unordered_map<std::string, std::unique_ptr<FileSystem>> registry_;
};

Status FileSystemRegistryImpl::Register(const std::string& scheme,
                                        FileSystemRegistry::Factory factory) {
  ScopedLocker<std::mutex> _(&mu_);
  registry_.emplace(scheme, std::unique_ptr<FileSystem>(factory()));
  return Status::OK();
}

FileSystem* FileSystemRegistryImpl::Lookup(const std::string& scheme) {
  ScopedLocker<std::mutex> _(&mu_);
  auto found = registry_.find(scheme);
  if (found == registry_.end()) {
    return nullptr;
  }
  return found->second.get();
}

Env::Env() {
  fs_registry_.reset(new FileSystemRegistryImpl);

  inter_tp_.reset(new ThreadPool(GLOBAL_FLAG(InterThreadNum)));
  inter_tp_->Startup();

  intra_tp_.reset(new ThreadPool(GLOBAL_FLAG(IntraThreadNum)));
  intra_tp_->Startup();

  // for system runtime, users can not reset this number
  reserved_tp_.reset(new ThreadPool(5));
  reserved_tp_->Startup();
}

Env::~Env() {
  if (inter_tp_) {
    inter_tp_->Shutdown();
  }

  if (intra_tp_) {
    intra_tp_->Shutdown();
  }

  if (reserved_tp_) {
    reserved_tp_->Shutdown();
  }
}

int32_t Env::GetServerId() {
  return GLOBAL_FLAG(ServerId);
}

int32_t Env::GetServerCount() {
  return GLOBAL_FLAG(ServerCount);
}

Status Env::GetFileSystem(const std::string& file_path,
                          FileSystem** result) {
  std::string scheme = io::GetScheme(file_path);
  FileSystem* file_system = fs_registry_->Lookup(scheme);
  if (!file_system) {
    LOG(ERROR) << "File system not implemented: " << file_path;
    return Status(error::NOT_FOUND, "File system not implemented");
  }
  *result = file_system;
  return Status::OK();
}

Status Env::RegisterFileSystem(
    const std::string& scheme,
    FileSystemRegistry::Factory factory) {
  return fs_registry_->Register(scheme, std::move(factory));
}

Env* Env::Default() {
  static Env* default_env = new Env;
  return default_env;
}

}  // namespace graphlearn
