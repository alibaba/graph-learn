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

#ifndef DGS_CORE_EXECUTION_OP_FACTORY_H_
#define DGS_CORE_EXECUTION_OP_FACTORY_H_

#include "core/execution/op_registry.h"

namespace dgs {
namespace execution {

class OpFactory {
public:
  static OpFactory* GetInstance() {
    static OpFactory factory;
    return &factory;
  }

  OpFactory(const OpFactory&) = delete;
  OpFactory& operator=(const OpFactory&) = delete;

  Op* Create(const std::string& op_name,
             OperatorId id, const Op::Params& params) {
    auto creator = op_registry_->Lookup(op_name);
    if (!creator) {
      LOG(ERROR) << "op " << op_name << " not registered.";
      return nullptr;
    }
    Op* op = (*creator)();
    op->SetParams(id, params);
    return op;
  }

private:
  OpFactory() : op_registry_(OpRegistry::GetInstance()) {}
  ~OpFactory() = default;

private:
  OpRegistry* op_registry_;
};

}  // namespace execution
}  // namespace dgs

#endif  // DGS_CORE_EXECUTION_OP_FACTORY_H_
