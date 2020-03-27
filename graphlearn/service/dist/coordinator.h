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

#ifndef GRAPHLEARN_SERVICE_DIST_COORDINATOR_H_
#define GRAPHLEARN_SERVICE_DIST_COORDINATOR_H_

#include <cstdint>
#include <string>
#include "graphlearn/include/status.h"

namespace graphlearn {

class Env;
class FileSystem;

/// A class to coordinate states among all servers. Because of the existence of
/// communication between servers, a coordinator is needed to manage the state
/// machine, making the distributed service behaives correctly.
class Coordinator {
public:
  /// Each server will hold a Coordinator client.
  Coordinator(int32_t server_id, int32_t server_count, Env* env);
  ~Coordinator();

  bool IsMaster() const;

  /// To tell that the current server started.
  Status Start();
  /// Whether all servers started.
  bool IsStartup() const;

  /// To tell that the current server was ready to work.
  Status SetReady();
  /// Whether all servers were ready.
  bool IsReady() const;

  /// To tell that the server received StopRequestPb from an user client.
  Status Stop(int32_t client_id, int32_t client_count);
  /// Whether all servers should stop.
  bool IsStopped() const;

private:
  void Refresh();
  void CheckStarted();
  void CheckReady();
  void CheckStopped();

  bool FileExist(const std::string& file_name);
  int32_t Counting(const std::string& sub_dir);
  Status Sink(const std::string& sub_dir, const std::string& file_name);

private:
  bool        started_;
  bool        ready_;
  bool        stopped_;
  int32_t     client_count_;
  int32_t     server_id_;
  int32_t     server_count_;
  std::string tracker_;
  FileSystem* fs_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_DIST_COORDINATOR_H_
