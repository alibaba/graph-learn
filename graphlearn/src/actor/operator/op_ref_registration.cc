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

#include "actor/generated/operator/graph_ops_ref.act.autogen.h"
#include "actor/generated/operator/sampler_ops_ref.act.autogen.h"

namespace graphlearn {
namespace act {
namespace op_ref_registration {

/// Graph Operators
OpRefRegistration<GetDegreeActor_ref> _degree_getter_("GetDegree");
OpRefRegistration<NodeGetterActor_ref> _node_getter_("GetNodes");
OpRefRegistration<LookupNodesActor_ref> _node_lookuper_("LookupNodes");
OpRefRegistration<EdgeGetterActor_ref> _edge_getter_("GetEdges");
OpRefRegistration<LookupEdgesActor_ref> _edge_lookuper_("LookupEdges");

/// Sampler Operators
OpRefRegistration<EdgeWeightSamplerActor_ref> _edge_weight_sampler_("EdgeWeightSampler");
OpRefRegistration<FullSamplerActor_ref> _full_sampler_("FullSampler");
OpRefRegistration<InDegreeNegativeSamplerActor_ref> _in_degree_negative_sampler_("InDegreeNegativeSampler");
OpRefRegistration<InDegreeSamplerActor_ref> _in_degree_sampler_("InDegreeSampler");
OpRefRegistration<NodeWeightNegativeSamplerActor_ref> _node_weight_negative_sampler_("NodeWeightNegativeSampler");
OpRefRegistration<RandomNegativeSamplerActor_ref> _random_negative_sampler_("RandomNegativeSampler");
OpRefRegistration<RandomSamplerActor_ref> _random_sampler_("RandomSampler");
OpRefRegistration<RandomWithoutReplacementSamplerActor_ref> _random_without_replacement_sampler_("RandomWithoutReplacementSampler");
OpRefRegistration<SoftInDegreeNegativeSamplerActor_ref> _soft_in_degree_negative_sampler_("SoftInDegreeNegativeSampler");
OpRefRegistration<TopkSamplerActor_ref> _topk_sampler_("TopkSampler");
OpRefRegistration<RandomWalkActor_ref> _random_walk_("RandomWalk");
OpRefRegistration<SubGraphSamplerActor_ref> _sub_graph_sampler_("SubGraphSampler");
}  // namespace op_ref_registration
}  // namespace act
}  // namespace graphlearn

