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

#ifndef GRAPHLEARN_INCLUDE_CONFIG_H_
#define GRAPHLEARN_INCLUDE_CONFIG_H_

#include <cstdint>
#include <string>

namespace graphlearn {

// Access global flag
#define GLOBAL_FLAG(name) ::graphlearn::g##name

// Declare kinds of global flag
#define DECLARE_GLOBAL_FLAG(name, type)  \
  extern type g##name;

#define DECLARE_INT32_GLOBAL_FLAG(name)  \
  DECLARE_GLOBAL_FLAG(name, int32_t)

#define DECLARE_INT64_GLOBAL_FLAG(name)  \
  DECLARE_GLOBAL_FLAG(name, int64_t)

#define DECLARE_FLOAT_GLOBAL_FLAG(name)  \
  DECLARE_GLOBAL_FLAG(name, float)

#define DECLARE_STRING_GLOBAL_FLAG(name) \
  DECLARE_GLOBAL_FLAG(name, std::string)

// Declare setters of global flag
#define DECLARE_SET_GLOBAL_FLAG(name, type)   \
  void SetGlobalFlag##name(type value);

#define DECLARE_SET_INT32_GLOBAL_FLAG(name)   \
  DECLARE_SET_GLOBAL_FLAG(name, int32_t)

#define DECLARE_SET_INT64_GLOBAL_FLAG(name)   \
  DECLARE_SET_GLOBAL_FLAG(name, int64_t)

#define DECLARE_SET_FLOAT_GLOBAL_FLAG(name)   \
  DECLARE_SET_GLOBAL_FLAG(name, float)

#define DECLARE_SET_STRING_GLOBAL_FLAG(name)  \
  DECLARE_SET_GLOBAL_FLAG(name, const std::string&)

// Declare the global flags
DECLARE_INT32_GLOBAL_FLAG(DeployMode)
DECLARE_INT32_GLOBAL_FLAG(ClientId)
DECLARE_INT32_GLOBAL_FLAG(ClientCount)
DECLARE_INT32_GLOBAL_FLAG(ServerId)
DECLARE_INT32_GLOBAL_FLAG(ServerCount)
DECLARE_INT32_GLOBAL_FLAG(Timeout)
DECLARE_INT32_GLOBAL_FLAG(RetryTimes)
DECLARE_INT32_GLOBAL_FLAG(InMemoryQueueSize)
DECLARE_INT32_GLOBAL_FLAG(DataInitBatchSize)
DECLARE_INT32_GLOBAL_FLAG(ShuffleBufferSize)
DECLARE_INT32_GLOBAL_FLAG(RpcMessageMaxSize)
DECLARE_INT32_GLOBAL_FLAG(InterThreadNum)
DECLARE_INT32_GLOBAL_FLAG(IntraThreadNum)
DECLARE_INT32_GLOBAL_FLAG(PartitionMode)
DECLARE_INT32_GLOBAL_FLAG(PaddingMode)
DECLARE_INT32_GLOBAL_FLAG(StorageMode)
DECLARE_INT32_GLOBAL_FLAG(TrackerMode)
DECLARE_INT64_GLOBAL_FLAG(AverageNodeCount)
DECLARE_INT64_GLOBAL_FLAG(AverageEdgeCount)
DECLARE_INT64_GLOBAL_FLAG(DefaultNeighborId)
DECLARE_INT64_GLOBAL_FLAG(DefaultIntAttribute)
DECLARE_FLOAT_GLOBAL_FLAG(DefaultFloatAttribute)
DECLARE_STRING_GLOBAL_FLAG(DefaultStringAttribute)
DECLARE_STRING_GLOBAL_FLAG(Tracker)
DECLARE_STRING_GLOBAL_FLAG(ServerHosts)

DECLARE_INT64_GLOBAL_FLAG(VineyardGraphID)
DECLARE_STRING_GLOBAL_FLAG(VineyardIPCSocket)
DECLARE_INT32_GLOBAL_FLAG(AttrStartIndex)
DECLARE_INT32_GLOBAL_FLAG(AttrEndIndex)

// Declare the setters
DECLARE_SET_INT32_GLOBAL_FLAG(DeployMode)
DECLARE_SET_INT32_GLOBAL_FLAG(ClientId)
DECLARE_SET_INT32_GLOBAL_FLAG(ClientCount)
DECLARE_SET_INT32_GLOBAL_FLAG(ServerId)
DECLARE_SET_INT32_GLOBAL_FLAG(ServerCount)
DECLARE_SET_INT32_GLOBAL_FLAG(Timeout)
DECLARE_SET_INT32_GLOBAL_FLAG(RetryTimes)
DECLARE_SET_INT32_GLOBAL_FLAG(InMemoryQueueSize)
DECLARE_SET_INT32_GLOBAL_FLAG(DataInitBatchSize)
DECLARE_SET_INT32_GLOBAL_FLAG(ShuffleBufferSize)
DECLARE_SET_INT32_GLOBAL_FLAG(RpcMessageMaxSize)
DECLARE_SET_INT32_GLOBAL_FLAG(InterThreadNum)
DECLARE_SET_INT32_GLOBAL_FLAG(IntraThreadNum)
DECLARE_SET_INT32_GLOBAL_FLAG(PartitionMode)
DECLARE_SET_INT32_GLOBAL_FLAG(PaddingMode)
DECLARE_SET_INT32_GLOBAL_FLAG(StorageMode)
DECLARE_SET_INT32_GLOBAL_FLAG(TrackerMode)
DECLARE_SET_INT64_GLOBAL_FLAG(AverageNodeCount)
DECLARE_SET_INT64_GLOBAL_FLAG(AverageEdgeCount)
DECLARE_SET_INT64_GLOBAL_FLAG(DefaultNeighborId)
DECLARE_SET_INT64_GLOBAL_FLAG(DefaultIntAttribute)
DECLARE_SET_FLOAT_GLOBAL_FLAG(DefaultFloatAttribute)
DECLARE_SET_STRING_GLOBAL_FLAG(DefaultStringAttribute)
DECLARE_SET_STRING_GLOBAL_FLAG(Tracker)
DECLARE_SET_STRING_GLOBAL_FLAG(ServerHosts)

DECLARE_SET_INT64_GLOBAL_FLAG(VineyardGraphID)
DECLARE_SET_STRING_GLOBAL_FLAG(VineyardIPCSocket)
DECLARE_SET_INT32_GLOBAL_FLAG(AttrStartIndex)
DECLARE_SET_INT32_GLOBAL_FLAG(AttrEndIndex)

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_CONFIG_H_
