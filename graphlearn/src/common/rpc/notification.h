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

#ifndef GRAPHLEARN_COMMON_RPC_NOTIFICATION_H_
#define GRAPHLEARN_COMMON_RPC_NOTIFICATION_H_

#include <functional>
#include <string>
#include "graphlearn/include/status.h"

namespace graphlearn {

class RpcNotificationImpl;

class RpcNotification {
public:
  typedef std::function<void(const std::string& req_type,
                             const Status& status)> Callback;

  RpcNotification();
  ~RpcNotification();

  // Set notification request type and reserve size.
  void Init(const std::string& req_type,
            int32_t size);

  // Add an rpc task to remote_id to tasks.
  // Returns:
  //   task id.
  int32_t AddRpcTask(int32_t remote_id);

  // Set notification callback.
  // This callback will be invoked when all rpcs are done or any rpc is failed.
  void SetCallback(Callback cb);

  // An rpc task to remote_id is finished.
  void Notify(int32_t remote_id);

  // An rpc task to remote_id is failed.
  void NotifyFail(int32_t remote_id, const Status& status);

  // Wait for sync notification.
  void Wait(int64_t timeout_ms = -1);

private:
  RpcNotificationImpl* impl_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_RPC_NOTIFICATION_H_
