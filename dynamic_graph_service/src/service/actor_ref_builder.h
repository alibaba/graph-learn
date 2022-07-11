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

#ifndef DGS_SERVICE_ACTOR_REF_BUILDER_H_
#define DGS_SERVICE_ACTOR_REF_BUILDER_H_

#include "service/generated/sampling_actor_ref.act.autogen.h"
#include "service/generated/data_update_actor_ref.act.autogen.h"
#include "service/generated/serving_actor_ref.act.autogen.h"

namespace dgs {

SamplingActor_ref
MakeSamplingActorInstRef(hiactor::scope_builder& builder);

SamplingActor_ref*
MakeSamplingActorInstRefPtr(hiactor::scope_builder& builder);

DataUpdateActor_ref
MakeDataUpdateActorInstRef(hiactor::scope_builder& builder);

ServingActor_ref
MakeServingActorInstRef(hiactor::scope_builder& builder);

}  // namespace dgs

#endif // DGS_SERVICE_ACTOR_REF_BUILDER_H_
