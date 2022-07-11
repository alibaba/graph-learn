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

#ifndef DGS_SERVICE_SERVING_GROUP_ACTG_H_
#define DGS_SERVICE_SERVING_GROUP_ACTG_H_

#include "hiactor/core/actor-template.hh"
#include "hiactor/core/reference_base.hh"

namespace dgs {

/// Customized serving actor group
class ANNOTATION(actor:group) serving_group : public hiactor::schedulable_actor_group {
public:
  serving_group(hiactor::actor_base* exec_ctx, const hiactor::byte_t* addr)
    : hiactor::schedulable_actor_group(exec_ctx, addr) {}

  bool compare(const actor_base* a, const actor_base* b) const override {
    return a->actor_id() < b->actor_id();
  }
};

hiactor::scope<serving_group> MakeServingGroupScope();

}  // namespace dgs

#endif  // DGS_SERVICE_SERVING_GROUP_ACTG_H_
