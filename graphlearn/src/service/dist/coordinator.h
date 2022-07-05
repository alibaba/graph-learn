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
#include <set>
#include <string>
#include <unordered_map>
#include "common/threading/sync/lock.h"
#include "include/constants.h"
#include "include/status.h"

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
  virtual ~Coordinator();

  virtual void Finallize();

  bool IsMaster() const;

  virtual Status Sync(const std::string& barrier) = 0;

  /// Do tell that the current server started.
  virtual Status Start() = 0;
  /// Set state of started for the current server.
  virtual Status SetStarted(int32_t server_id = -1) = 0;
  /// Is global state of all servers started or not.
  bool IsStartup() const;

  /// Do tell that the current server has initialized the data.
  virtual Status Init() = 0;
  /// Set state of inited for the current server.
  virtual Status SetInited(int32_t server_id = -1) = 0;
  /// Is global state of all servers inited or not.
  bool IsInited() const;

  /// Do tell that the current server was ready to work.
  virtual Status Prepare() = 0;
  /// Set state of ready for the current server.
  virtual Status SetReady(int32_t server_id = -1) = 0;
  /// Is global state of all servers ready or not.
  bool IsReady() const;

  /// Do tell that the current server received stop request from a client.
  virtual Status Stop(int32_t client_id, int32_t client_count) = 0;
  /// Set state of stopped for the current client.
  virtual Status SetStopped(int32_t client_id = -1,
                            int32_t client_count = 0) = 0;
  /// Is global state of all clients stopped or not.
  bool IsStopped() const;

  virtual Status SetState(int32_t state, int32_t server_id = -1) = 0;

protected:
  virtual void Refresh();
  virtual void CheckStarted() = 0;
  virtual void CheckInited() = 0;
  virtual void CheckReady() = 0;
  virtual void CheckStopped() = 0;

protected:
  int32_t     client_count_;
  int32_t     server_id_;
  int32_t     server_count_;
  SystemState state_;
};

class FSCoordinator : public Coordinator {
public:
  FSCoordinator(int32_t server_id, int32_t server_count, Env* env);
  ~FSCoordinator() = default;

  void Finallize() override;

  Status Sync(const std::string& barrier) override;

  Status Start() override;
  Status SetStarted(int32_t server_id = -1) override;

  Status Init() override;
  Status SetInited(int32_t server_id = -1) override;

  Status Prepare() override;
  Status SetReady(int32_t server_id = -1) override;

  Status Stop(int32_t client_id, int32_t client_count) override;
  Status SetStopped(int32_t client_id = -1,
                    int32_t client_count = 0) override;

  Status SetState(int32_t state, int32_t id) override { return Status::OK(); }

private:
  void Refresh() override;
  void CheckStarted() override;
  void CheckInited() override;
  void CheckReady() override;
  void CheckStopped() override;

  bool IsReady(const std::string& barrier);
  bool FileExist(const std::string& file_name);
  int32_t Counting(const std::string& sub_dir);
  Status Sink(const std::string& sub_dir, const std::string& file_name);

private:
  std::string tracker_;
  FileSystem* fs_;
};

class RPCCoordinator : public Coordinator {
public:
  RPCCoordinator(int32_t server_id, int32_t server_count, Env* env);
  ~RPCCoordinator() = default;

  Status Sync(const std::string& barrier) override;

  Status Start() override;
  Status SetStarted(int32_t server_id = -1) override;

  Status Init() override;
  Status SetInited(int32_t server_id = -1) override;

  Status Prepare() override;
  Status SetReady(int32_t server_id = -1) override;

  Status Stop(int32_t client_id, int32_t client_count) override;
  Status SetStopped(int32_t client_id = -1,
                    int32_t client_count = 0) override;
  Status SetState(int32_t state, int32_t id) override;

private:
  void Refresh() override;
  void CheckStarted() override;
  void CheckInited() override;
  void CheckReady() override;
  void CheckStopped() override;

  Status SetState(SystemState state, int32_t id);
  void CheckState(SystemState state, int32_t count);
  void CheckState(int32_t state, int32_t count);
  Status ReportState(int32_t target, int32_t state,
                     int32_t id = -1, int32_t count = -1);

private:
  std::mutex mtx_;
  int32_t reserved_state_;
  std::unordered_map<int32_t, std::set<int32_t>> state_map_;
};

Coordinator* GetCoordinator(int32_t server_id, int32_t server_count,
                            Env* env);

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_DIST_COORDINATOR_H_
