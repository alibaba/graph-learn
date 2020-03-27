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

#ifndef GRAPHLEARN_PLATFORM_ENV_H_
#define GRAPHLEARN_PLATFORM_ENV_H_

#include <memory>
#include <string>
#include "graphlearn/common/threading/runner/threadpool.h"
#include "graphlearn/platform/file_system.h"

namespace graphlearn {

class Env {
public:
  ~Env();
  static Env* Default();

  int32_t GetServerId();
  int32_t GetServerCount();

  Status GetFileSystem(
      const std::string& file_path,
      FileSystem** result);

  Status RegisterFileSystem(
      const std::string& scheme,
      FileSystemRegistry::Factory factory);

  ThreadPool* InterThreadPool() {
    return inter_tp_.get();
  }

  ThreadPool* IntraThreadPool() {
    return intra_tp_.get();
  }

  ThreadPool* ReservedThreadPool() {
    return reserved_tp_.get();
  }

private:
  Env();

private:
  std::unique_ptr<FileSystemRegistry> fs_registry_;
  std::unique_ptr<ThreadPool>         inter_tp_;
  std::unique_ptr<ThreadPool>         intra_tp_;
  std::unique_ptr<ThreadPool>         reserved_tp_;
};

namespace register_file_system {

template <typename Factory>
struct Register {
  Register(Env* env, const std::string& scheme) {
    env->RegisterFileSystem(scheme,
                            []() -> FileSystem* { return new Factory; });
  }
};

}  // namespace register_file_system

// Register a FileSystem implementation for a scheme.
// Files whose name start with "scheme://" will be routed to
// use this implementation.
#define REGISTER_FILE_SYSTEM_ENV(env, scheme, factory) \
  static ::graphlearn::register_file_system::Register<factory> \
      register_##factory =  \
          ::graphlearn::register_file_system::Register<factory>(env, scheme)

#define REGISTER_FILE_SYSTEM(scheme, factory) \
  REGISTER_FILE_SYSTEM_ENV(Env::Default(), scheme, factory);

}  // namespace graphlearn

#endif  // GRAPHLEARN_PLATFORM_ENV_H_
