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

#include "actor/operator/op_ref_factory.h"

#include "actor/generated/edge_getter_ref.act.autogen.h"
#include "actor/generated/edge_lookuper_ref.act.autogen.h"
#include "actor/generated/edge_weight_sampler_ref.act.autogen.h"
#include "actor/generated/full_sampler_ref.act.autogen.h"
#include "actor/generated/in_degree_negative_sampler_ref.act.autogen.h"
#include "actor/generated/in_degree_sampler_ref.act.autogen.h"
#include "actor/generated/node_getter_ref.act.autogen.h"
#include "actor/generated/node_lookuper_ref.act.autogen.h"
#include "actor/generated/node_weight_negative_sampler_ref.act.autogen.h"
#include "actor/generated/random_negative_sampler_ref.act.autogen.h"
#include "actor/generated/random_sampler_ref.act.autogen.h"
#include "actor/generated/random_without_replacement_sampler_ref.act.autogen.h"
#include "actor/generated/soft_in_degree_negative_sampler_ref.act.autogen.h"
#include "actor/generated/topk_sampler_ref.act.autogen.h"

namespace graphlearn {
namespace act {
namespace ref_registration {

OpRefRegistration<EdgeGetterActor_ref> _edge_getter_("GetEdges");
OpRefRegistration<EdgeLookuperActor_ref> _edge_lookuper_("LookupEdges");
OpRefRegistration<EdgeWeightSamplerActor_ref> _edge_weight_sampler_("EdgeWeightSampler");
OpRefRegistration<FullSamplerActor_ref> _full_sampler_("FullSampler");
OpRefRegistration<InDegreeNegativeSamplerActor_ref> _in_degree_negative_sampler_("InDegreeNegativeSampler");
OpRefRegistration<InDegreeSamplerActor_ref> _in_degree_sampler_("InDegreeSampler");
OpRefRegistration<NodeGetterActor_ref> _node_getter_("GetNodes");
OpRefRegistration<NodeLookuperActor_ref> _node_lookuper_("LookupNodes");
OpRefRegistration<NodeWeightNegativeSamplerActor_ref> _node_weight_negative_sampler_("NodeWeightNegativeSampler");
OpRefRegistration<RandomNegativeSamplerActor_ref> _random_negative_sampler_("RandomNegativeSampler");
OpRefRegistration<RandomSamplerActor_ref> _random_sampler_("RandomSampler");
OpRefRegistration<RandomWithoutReplacementSamplerActor_ref> _random_without_replacement_sampler_("RandomWithoutReplacementSampler");
OpRefRegistration<SoftInDegreeNegativeSamplerActor_ref> _soft_in_degree_negative_sampler_("SoftInDegreeNegativeSampler");
OpRefRegistration<TopKSamplerActor_ref> _topK_sampler_("TopkSampler");

}  // namespace ref_registration
}  // namespace act
}  // namespace graphlearn

