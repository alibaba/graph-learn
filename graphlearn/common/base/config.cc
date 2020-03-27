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

#include "graphlearn/include/config.h"

#include <algorithm>
#include <cstring>
#include "graphlearn/common/base/log.h"

namespace graphlearn {

// Define kinds of global flag
#define DEFINE_GLOBAL_FLAG(name, type, value)  \
  type g##name = value;

#define DEFINE_INT32_GLOBAL_FLAG(name, value)  \
  DEFINE_GLOBAL_FLAG(name, int32_t, value)

#define DEFINE_INT64_GLOBAL_FLAG(name, value)  \
  DEFINE_GLOBAL_FLAG(name, int64_t, value)

#define DEFINE_FLOAT_GLOBAL_FLAG(name, value)  \
  DEFINE_GLOBAL_FLAG(name, float, value)

#define DEFINE_STRING_GLOBAL_FLAG(name, value) \
  DEFINE_GLOBAL_FLAG(name, std::string, value)

// Define the setters of global flag
#define DEFINE_SET_GLOBAL_FLAG(name, type)   \
void SetGlobalFlag##name(type value) {       \
  GLOBAL_FLAG(name) = value;                 \
}

#define DEFINE_SET_INT32_GLOBAL_FLAG(name)   \
  DEFINE_SET_GLOBAL_FLAG(name, int32_t)

#define DEFINE_SET_INT64_GLOBAL_FLAG(name)   \
  DEFINE_SET_GLOBAL_FLAG(name, int64_t)

#define DEFINE_SET_FLOAT_GLOBAL_FLAG(name)   \
  DEFINE_SET_GLOBAL_FLAG(name, float)

#define DEFINE_SET_STRING_GLOBAL_FLAG(name)  \
  DEFINE_SET_GLOBAL_FLAG(name, const std::string&)

// Define the global flags
DEFINE_INT32_GLOBAL_FLAG(DeployMode, 0)
DEFINE_INT32_GLOBAL_FLAG(ClientId, 0)
DEFINE_INT32_GLOBAL_FLAG(ClientCount, 1)
DEFINE_INT32_GLOBAL_FLAG(ServerId, 0)
DEFINE_INT32_GLOBAL_FLAG(ServerCount, 1)
DEFINE_INT32_GLOBAL_FLAG(RetryTimes, 10)
DEFINE_INT32_GLOBAL_FLAG(InMemoryQueueSize, 10240)
DEFINE_INT32_GLOBAL_FLAG(DataInitBatchSize, 10240)
DEFINE_INT32_GLOBAL_FLAG(ShuffleBufferSize, 10240)
DEFINE_INT32_GLOBAL_FLAG(RpcMessageMaxSize, 16 * 1024 * 1024)
DEFINE_INT32_GLOBAL_FLAG(InterThreadNum, 32)
DEFINE_INT32_GLOBAL_FLAG(IntraThreadNum, 32)
DEFINE_INT32_GLOBAL_FLAG(PartitionMode, 1)
DEFINE_INT64_GLOBAL_FLAG(AverageNodeCount, 10000)
DEFINE_INT64_GLOBAL_FLAG(AverageEdgeCount, 10000)
DEFINE_INT64_GLOBAL_FLAG(DefaultNeighborId, 0)
DEFINE_INT64_GLOBAL_FLAG(DefaultIntAttribute, 0)
DEFINE_FLOAT_GLOBAL_FLAG(DefaultFloatAttribute, 0.0)
DEFINE_STRING_GLOBAL_FLAG(DefaultStringAttribute, "")
DEFINE_STRING_GLOBAL_FLAG(Tracker, "/tmp/graphlearn/")

// Define the setters
DEFINE_SET_INT32_GLOBAL_FLAG(DeployMode)
DEFINE_SET_INT32_GLOBAL_FLAG(ClientId)
DEFINE_SET_INT32_GLOBAL_FLAG(ClientCount)
DEFINE_SET_INT32_GLOBAL_FLAG(ServerId)
DEFINE_SET_INT32_GLOBAL_FLAG(ServerCount)
DEFINE_SET_INT32_GLOBAL_FLAG(RetryTimes)
DEFINE_SET_INT32_GLOBAL_FLAG(InMemoryQueueSize)
DEFINE_SET_INT32_GLOBAL_FLAG(DataInitBatchSize)
DEFINE_SET_INT32_GLOBAL_FLAG(ShuffleBufferSize)
DEFINE_SET_INT32_GLOBAL_FLAG(RpcMessageMaxSize)
DEFINE_SET_INT32_GLOBAL_FLAG(InterThreadNum)
DEFINE_SET_INT32_GLOBAL_FLAG(IntraThreadNum)
DEFINE_SET_INT32_GLOBAL_FLAG(PartitionMode)
DEFINE_SET_INT64_GLOBAL_FLAG(AverageNodeCount)
DEFINE_SET_INT64_GLOBAL_FLAG(AverageEdgeCount)
DEFINE_SET_INT64_GLOBAL_FLAG(DefaultNeighborId)
DEFINE_SET_INT64_GLOBAL_FLAG(DefaultIntAttribute)
DEFINE_SET_FLOAT_GLOBAL_FLAG(DefaultFloatAttribute)
DEFINE_SET_STRING_GLOBAL_FLAG(DefaultStringAttribute)
DEFINE_SET_STRING_GLOBAL_FLAG(Tracker)

}  // namespace graphlearn
