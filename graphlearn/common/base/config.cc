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
#include <limits>
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

// Define the getters of global flag
#define DEFINE_GET_GLOBAL_FLAG(name, type)         \
type GetGlobalFlag##name() {                 \
  return GLOBAL_FLAG(name);                  \
}

#define DEFINE_GET_INT32_GLOBAL_FLAG(name)   \
  DEFINE_GET_GLOBAL_FLAG(name, int32_t)

#define DEFINE_GET_INT64_GLOBAL_FLAG(name)   \
  DEFINE_GET_GLOBAL_FLAG(name, int64_t)

#define DEFINE_GET_FLOAT_GLOBAL_FLAG(name)   \
  DEFINE_GET_GLOBAL_FLAG(name, float)

#define DEFINE_GET_STRING_GLOBAL_FLAG(name)  \
  DEFINE_GET_GLOBAL_FLAG(name, const std::string&)


// Define the global flags
DEFINE_INT32_GLOBAL_FLAG(DeployMode, 0)
DEFINE_INT32_GLOBAL_FLAG(EnableActor, 0)
DEFINE_INT32_GLOBAL_FLAG(ClientId, 0)
DEFINE_INT32_GLOBAL_FLAG(ClientCount, 1)
DEFINE_INT32_GLOBAL_FLAG(ServerId, 0)
DEFINE_INT32_GLOBAL_FLAG(ServerCount, 1)
DEFINE_INT32_GLOBAL_FLAG(Timeout, 60)
DEFINE_INT32_GLOBAL_FLAG(RetryTimes, 10)
DEFINE_INT32_GLOBAL_FLAG(InMemoryQueueSize, 10240)
DEFINE_INT32_GLOBAL_FLAG(TapeCapacity, 10)
DEFINE_INT32_GLOBAL_FLAG(DatasetCapacity, 10)
DEFINE_INT32_GLOBAL_FLAG(DataInitBatchSize, 10240)
DEFINE_INT32_GLOBAL_FLAG(ShuffleBufferSize, 10240)
DEFINE_INT32_GLOBAL_FLAG(RpcMessageMaxSize, std::numeric_limits<int32_t>::max())
DEFINE_INT32_GLOBAL_FLAG(InterThreadNum, 32)
DEFINE_INT32_GLOBAL_FLAG(IntraThreadNum, 32)
DEFINE_INT32_GLOBAL_FLAG(PartitionMode, 1)
DEFINE_INT32_GLOBAL_FLAG(StorageMode, 2)
DEFINE_INT32_GLOBAL_FLAG(PaddingMode, 1) // 0: replic, 1: circular
DEFINE_INT32_GLOBAL_FLAG(TrackerMode, 1)  // 0: Rpc, 1: FileSystem
DEFINE_INT64_GLOBAL_FLAG(AverageNodeCount, 10000)
DEFINE_INT64_GLOBAL_FLAG(AverageEdgeCount, 10000)
DEFINE_INT64_GLOBAL_FLAG(DefaultNeighborId, 0)
DEFINE_INT64_GLOBAL_FLAG(DefaultIntAttribute, 0)
DEFINE_FLOAT_GLOBAL_FLAG(DefaultFloatAttribute, 0.0)
DEFINE_STRING_GLOBAL_FLAG(DefaultStringAttribute, "")
DEFINE_STRING_GLOBAL_FLAG(Tracker, "/tmp/graphlearn/")
DEFINE_STRING_GLOBAL_FLAG(ServerHosts, "")
DEFINE_INT32_GLOBAL_FLAG(KnnMetric, 0)  // 0 is l2, 1 is inner product.
DEFINE_INT32_GLOBAL_FLAG(NegativeSamplingRetryTimes, 5)
DEFINE_INT32_GLOBAL_FLAG(IgnoreInvalid, 1) // 1 is True, 0 is False.
DEFINE_INT32_GLOBAL_FLAG(LocalShardCount, 8)


// Define the setters
DEFINE_SET_INT32_GLOBAL_FLAG(DeployMode)
DEFINE_SET_INT32_GLOBAL_FLAG(EnableActor)
DEFINE_SET_INT32_GLOBAL_FLAG(ClientId)
DEFINE_SET_INT32_GLOBAL_FLAG(ClientCount)
DEFINE_SET_INT32_GLOBAL_FLAG(ServerId)
DEFINE_SET_INT32_GLOBAL_FLAG(ServerCount)
DEFINE_SET_INT32_GLOBAL_FLAG(Timeout)
DEFINE_SET_INT32_GLOBAL_FLAG(RetryTimes)
DEFINE_SET_INT32_GLOBAL_FLAG(InMemoryQueueSize)
DEFINE_SET_INT32_GLOBAL_FLAG(TapeCapacity)
DEFINE_SET_INT32_GLOBAL_FLAG(DatasetCapacity)
DEFINE_SET_INT32_GLOBAL_FLAG(DataInitBatchSize)
DEFINE_SET_INT32_GLOBAL_FLAG(ShuffleBufferSize)
DEFINE_SET_INT32_GLOBAL_FLAG(RpcMessageMaxSize)
DEFINE_SET_INT32_GLOBAL_FLAG(InterThreadNum)
DEFINE_SET_INT32_GLOBAL_FLAG(IntraThreadNum)
DEFINE_SET_INT32_GLOBAL_FLAG(PartitionMode)
DEFINE_SET_INT32_GLOBAL_FLAG(StorageMode)
DEFINE_SET_INT32_GLOBAL_FLAG(PaddingMode)
DEFINE_SET_INT32_GLOBAL_FLAG(TrackerMode)
DEFINE_SET_INT64_GLOBAL_FLAG(AverageNodeCount)
DEFINE_SET_INT64_GLOBAL_FLAG(AverageEdgeCount)
DEFINE_SET_INT64_GLOBAL_FLAG(DefaultNeighborId)
DEFINE_SET_INT64_GLOBAL_FLAG(DefaultIntAttribute)
DEFINE_SET_FLOAT_GLOBAL_FLAG(DefaultFloatAttribute)
DEFINE_SET_STRING_GLOBAL_FLAG(DefaultStringAttribute)
DEFINE_SET_STRING_GLOBAL_FLAG(Tracker)
DEFINE_SET_STRING_GLOBAL_FLAG(ServerHosts)
DEFINE_SET_INT32_GLOBAL_FLAG(KnnMetric)
DEFINE_SET_INT32_GLOBAL_FLAG(NegativeSamplingRetryTimes)
DEFINE_SET_INT32_GLOBAL_FLAG(IgnoreInvalid)
DEFINE_SET_INT32_GLOBAL_FLAG(LocalShardCount)

// Define the getters
/// Only export flags that are needed by system.
DEFINE_GET_INT32_GLOBAL_FLAG(TrackerMode)

}  // namespace graphlearn
