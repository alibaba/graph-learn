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

#ifndef GRAPHLEARN_ACTOR_GRAPH_GRAPH_ACTOR_ACT_H_
#define GRAPHLEARN_ACTOR_GRAPH_GRAPH_ACTOR_ACT_H_

#include <cstdint>
#include "brane/actor/actor_implementation.hh"
#include "brane/actor/reference_base.hh"
#include "brane/util/common-utils.hh"
#include "brane/util/data_type.hh"
#include "actor/graph/control_actor.act.h"
#include "actor/graph/wrapper_request.h"

namespace graphlearn {

class GraphStore;

namespace actor {

class ANNOTATION(actor:reference) GraphActorRef
  : public brane::reference_base {
public:
  void UpdateNodes(UpdateNodesRequestWrapper&& request);
  void UpdateEdges(UpdateEdgesRequestWrapper&& request);
  void ReceiveEOS();

  // Constructor.
  ACTOR_ITFC_CTOR(GraphActorRef);
  // Destructor
  ACTOR_ITFC_DTOR(GraphActorRef);
};

class ANNOTATION(actor:implement) GraphActor
  : public brane::stateful_actor {
public:
  void UpdateNodes(UpdateNodesRequestWrapper&& request);
  void UpdateEdges(UpdateEdgesRequestWrapper&& request);
  void ReceiveEOS();

  // Constructor.
  ACTOR_IMPL_CTOR(GraphActor);
  // Destructor.
  ACTOR_IMPL_DTOR(GraphActor);
  // Do work
  ACTOR_DO_WORK() override;

private:
  int         received_eos_number_;
  int         nodes_num_ = 0;
  int         edges_num_ = 0;
  int         data_parser_num_;
  GraphStore* store_;
};

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_GRAPH_ACTOR_ACT_H_
