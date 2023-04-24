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

#ifndef GRAPHLEARN_ACTOR_OPERATOR_SAMPLER_OPS_ACT_H_
#define GRAPHLEARN_ACTOR_OPERATOR_SAMPLER_OPS_ACT_H_

#include "actor/operator/op_def_macro.h"
#include "include/sampling_request.h"
#include "include/random_walk_request.h"
#include "include/subgraph_request.h"

namespace graphlearn {
namespace act {

DEFINE_OP_ACTOR(EdgeWeightSampler, SamplingRequest, SamplingResponse)
DEFINE_OP_ACTOR(FullSampler, SamplingRequest, SamplingResponse)
DEFINE_OP_ACTOR(InDegreeNegativeSampler, SamplingRequest, SamplingResponse)
DEFINE_OP_ACTOR(InDegreeSampler, SamplingRequest, SamplingResponse)
DEFINE_OP_ACTOR(NodeWeightNegativeSampler, SamplingRequest, SamplingResponse)
DEFINE_OP_ACTOR(RandomNegativeSampler, SamplingRequest, SamplingResponse)
DEFINE_OP_ACTOR(RandomSampler, SamplingRequest, SamplingResponse)
DEFINE_OP_ACTOR(RandomWithoutReplacementSampler, SamplingRequest, SamplingResponse)
DEFINE_OP_ACTOR(SoftInDegreeNegativeSampler, SamplingRequest, SamplingResponse)
DEFINE_OP_ACTOR(TopkSampler, SamplingRequest, SamplingResponse)
DEFINE_OP_ACTOR(RandomWalk, RandomWalkRequest, RandomWalkResponse)
DEFINE_OP_ACTOR(SubGraphSampler, SubGraphRequest, SubGraphResponse)

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_OPERATOR_SAMPLER_OPS_ACT_H_
