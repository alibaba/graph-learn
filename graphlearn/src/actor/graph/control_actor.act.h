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

#ifndef GRAPHLEARN_ACTOR_GRAPH_CONTROL_ACTOR_ACT_H_
#define GRAPHLEARN_ACTOR_GRAPH_CONTROL_ACTOR_ACT_H_

#include <cstdint>
#include <memory>

#include "brane/actor/actor_implementation.hh"
#include "brane/actor/reference_base.hh"
#include "brane/util/common-utils.hh"
#include "brane/util/data_type.hh"

namespace graphlearn {
namespace actor {

class ANNOTATION(actor:reference) ControlActorRef
  : public brane::reference_base {
public:
  void ReceiveEOS();
  void StopActor();
  // Constructor.
  ACTOR_ITFC_CTOR(ControlActorRef);
  // Destructor
  ACTOR_ITFC_DTOR(ControlActorRef);
};

class ANNOTATION(actor:implement) ControlActor
  : public brane::stateful_actor {
public:
  void ReceiveEOS();
  void StopActor();

  // Constructor.
  ACTOR_IMPL_CTOR(ControlActor);
  // Destructor.
  ACTOR_IMPL_DTOR(ControlActor);
  // Do work
  ACTOR_DO_WORK() override;

private:
  int received_eos_number_;
};

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_CONTROL_ACTOR_ACT_H_
