/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef DGS_COMMON_CONSTANTS_H_
#define DGS_COMMON_CONSTANTS_H_

#include <cstdint>

namespace dgs {

static constexpr uint32_t kFrontierResolutionInHour = 1;

namespace act {

static constexpr uint32_t kSamplingActorInstId = 0;

static constexpr uint32_t kDataUpdateActorInstId = 0;
static constexpr uint32_t kServingActorInstId = 1;

}  // namespace act

}  // namespace dgs

#endif  // DGS_COMMON_CONSTANTS_H_
