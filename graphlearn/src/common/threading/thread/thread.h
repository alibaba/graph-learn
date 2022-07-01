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

#ifndef GRAPHLEARN_COMMON_THREADING_THREAD_THREAD_H_
#define GRAPHLEARN_COMMON_THREADING_THREAD_THREAD_H_

#include <pthread.h>
#include "graphlearn/common/base/closure.h"
#include "graphlearn/common/threading/sync/waitable_event.h"
#include "graphlearn/platform/protobuf.h"

namespace graphlearn {

using ThreadHandle = pthread_t;

ThreadHandle CreateThread(::PB_NAMESPACE::Closure* func,
                          WaitableEvent* wait = nullptr,
                          const char* name = nullptr);

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_THREADING_THREAD_THREAD_H_
