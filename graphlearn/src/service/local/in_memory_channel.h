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

#ifndef GRAPHLEARN_SERVICE_LOCAL_IN_MEMORY_CHANNEL_H_
#define GRAPHLEARN_SERVICE_LOCAL_IN_MEMORY_CHANNEL_H_

#include "service/call.h"

namespace graphlearn {

template<typename T>
class EventQueue;

class InMemoryChannel {
public:
  explicit InMemoryChannel(EventQueue<Call*>* queue) : queue_(queue) {
  }

  ~InMemoryChannel() = default;

  void CallMethod(uint16_t method_id,
                  const BaseRequest* request,
                  BaseResponse* response,
                  StatusWrapper* status);

private:
  EventQueue<Call*>* queue_;  // not owned
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_LOCAL_IN_MEMORY_CHANNEL_H_

