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

#ifndef GRAPHLEARN_ACTOR_OPERATOR_OP_DEF_MACRO_ACT_H_
#define GRAPHLEARN_ACTOR_OPERATOR_OP_DEF_MACRO_ACT_H_

#include "actor/operator/base_op.act.h"

namespace graphlearn {
namespace act {

#define DEFINE_OP_ACTOR(name, request, response)  \
  class ANNOTATION(actor:impl) name##Actor : public BaseOperatorActor {  \
  public:  \
    name##Actor(hiactor::actor_base* exec_ctx, const hiactor::byte_t* addr)  \
        : BaseOperatorActor(exec_ctx, addr) {  \
      set_max_concurrency(UINT32_MAX);  \
      SetOp(#name);  \
    }  \
    ~name##Actor() override = default;  \
    seastar::future<TensorMap> ANNOTATION(actor:method) Process(TensorMap&& tensors) override { \
      request req;  \
      req.Init(*params_);  \
      req.Set(tensors.tensors_);  \
      response res;  \
      impl_->Process(&req, &res);  \
      return seastar::make_ready_future<TensorMap>(std::move(res.tensors_));  \
    }  \
    ACTOR_DO_WORK()  \
  };

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_OPERATOR_OP_DEF_MACRO_ACT_H_
