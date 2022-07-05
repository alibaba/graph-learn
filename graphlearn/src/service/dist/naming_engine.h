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

#ifndef GRAPHLEARN_SERVICE_DIST_NAMING_ENGINE_H_
#define GRAPHLEARN_SERVICE_DIST_NAMING_ENGINE_H_

#include <cstdint>
#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include <vector>
#include "include/config.h"
#include "include/status.h"

namespace graphlearn {

class FileSystem;

/// A class to implement address keeping. Each server instance
/// can register its listening endpoint to NamingEngine, and also
/// can get endpoints of the other servers.
class NamingEngine {
public:
  static NamingEngine* GetInstance();
  virtual ~NamingEngine() = default;

  void SetCapacity(int32_t capacity);

  /// Return the registered count, which may be changed in realtime.
  int32_t Size() const;

  /// Return the endpoint corresponding with the server id.
  /// If not found or the server was not ready, return empty string.
  std::string Get(int32_t server_id);

  virtual Status Update(const std::vector<std::string>& endpoints);

  /// Tell the engine to update the server endpoint. This method is
  /// called by each distributed server instance to sync their endpoints.
  virtual Status Update(int32_t server_id, const std::string& endpoint);

  /// Stop the engine.
  virtual void Stop() {}

protected:
  NamingEngine();

protected:
  std::mutex  mtx_;
  int32_t     size_;
  std::vector<std::string> endpoints_;
};

class FSNamingEngine : public NamingEngine {
public:
  Status Update(int32_t server_id, const std::string& endpoint) override;
  /// Stop the background refresh thread.
  void Stop() override;

public:
  explicit FSNamingEngine(std::string&& prefix = "");
  ~FSNamingEngine();

private:
  void Refresh();
  void Parse(const std::vector<std::string>& names);

private:
  std::string tracker_;
  FileSystem* fs_;
  volatile bool stopping_;
  volatile bool stopped_;
};

class SpecNamingEngine : public NamingEngine {
public:
  SpecNamingEngine() : NamingEngine() {
    endpoints_.resize(GLOBAL_FLAG(ServerCount));
  }
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_DIST_NAMING_ENGINE_H_

