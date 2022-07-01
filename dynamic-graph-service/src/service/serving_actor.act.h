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

#ifndef DGS_SERVICE_SERVING_ACTOR_ACT_H_
#define DGS_SERVICE_SERVING_ACTOR_ACT_H_

#include "hiactor/core/actor-template.hh"
#include "hiactor/core/reference_base.hh"

#include "core/storage/sample_store.h"
#include "service/request/admin_request.h"
#include "service/request/query_request.h"
#include "service/request/query_response.h"

namespace dgs {

class ANNOTATION(actor:impl) ServingActor
  : public hiactor::stateful_actor {
public:
  ServingActor(hiactor::actor_base* exec_ctx, const hiactor::byte_t* addr);
  ~ServingActor() override = default;

  seastar::future<QueryResponse> ANNOTATION(actor:method)
  RunQuery(RunQueryRequest&& req);

  seastar::future<actor::Void> ANNOTATION(actor:method)
  ExecuteAdminOperation(AdminRequest&& req);

  // Do work
  ACTOR_DO_WORK()

private:
  void InitializeImpl(const AdminRequest& req);
  void PauseExecutionImpl(const AdminRequest& req) {}
  void ResumeExecutionImpl(const AdminRequest& req) {}

private:
  storage::SampleStore*           sample_store_;
  std::unique_ptr<execution::Dag> dag_;
};

}  // namespace dgs

#endif  // DGS_SERVICE_SERVING_ACTOR_ACT_H_
