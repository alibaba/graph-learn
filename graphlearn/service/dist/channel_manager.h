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

#ifndef GRAPHLEARN_SERVICE_DIST_CHANNEL_MANAGER_H_
#define GRAPHLEARN_SERVICE_DIST_CHANNEL_MANAGER_H_

#include <cstdint>
#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include <vector>
#include "graphlearn/include/status.h"
#include "graphlearn/service/dist/grpc_channel.h"

namespace graphlearn {

class NamingEngine;
class LoadBalancer;

class ChannelManager {
public:
  static ChannelManager* GetInstance();
  ~ChannelManager();

  void SetCapacity(int32_t capacity);
  std::vector<int32_t> GetOwnServers();

  /// Stop the background refresh thread.
  /// Be sure that ChannelManager shoud stop after NamingEngine.
  void Stop();

  GrpcChannel* ConnectTo(int32_t server_id);
  GrpcChannel* AutoSelect();

private:
  ChannelManager();

  std::string GetEndpoint(int32_t server_id);
  void Refresh();

private:
  std::mutex    mtx_;
  std::atomic<bool> stopped_;
  NamingEngine* engine_;
  LoadBalancer* balancer_;
  std::vector<GrpcChannel*> channels_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_DIST_CHANNEL_MANAGER_H_

