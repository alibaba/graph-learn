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

#include "graphlearn/core/operator/utils/get_node_attributes_wrapper.h"

#include "graphlearn/include/config.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/operator/op_factory.h"
#include "graphlearn/core/runner/op_runner.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {
namespace op {

GetNodeAttributesWrapper::GetNodeAttributesWrapper(
    const std::string& node_type,
    const int64_t* id,
    int32_t batch_size)
  : res_(nullptr), req_(nullptr), i_cur_(0), i_num_(0),
    f_cur_(0), f_num_(0), s_cur_(0), s_num_(0) {
  status_ = this->Lookup(node_type, id, batch_size);
}

GetNodeAttributesWrapper::~GetNodeAttributesWrapper() {
  delete res_;
  delete req_;
}

Status GetNodeAttributesWrapper::Lookup(const std::string& node_type,
                                        const int64_t* id,
                                        int32_t batch_size) {
  req_ = new LookupNodesRequest(node_type);
  req_->Set(id, batch_size);

  res_ = new LookupNodesResponse();

  Operator* op = OpFactory::GetInstance()->Create("LookupNodes");
  std::unique_ptr<OpRunner> runner = GetOpRunner(Env::Default(), op);
  Status s = runner->Run(req_, res_);

  if (!s.ok()) {
    LOG(ERROR) << "GetNodeAttributesWrapper get failed" << ":" << s.ToString();
  }
  i_num_ = res_->IntAttrNum();
  f_num_ = res_->FloatAttrNum();
  s_num_ = res_->StringAttrNum();
  return s;
}

const Status& GetNodeAttributesWrapper::GetStatus() {
  return status_;
}

const int64_t* GetNodeAttributesWrapper::NextIntAttrs() {
  if (i_num_ <= 0) {
    return nullptr;
  }
  return res_->IntAttrs() + (i_cur_++ * i_num_);
}

const float* GetNodeAttributesWrapper::NextFloatAttrs() {
  if (f_num_ <= 0) {
    return nullptr;
  }
  return res_->FloatAttrs() + (f_cur_++ * f_num_);
}

const std::string* const* GetNodeAttributesWrapper::NextStrAttrs() {
  if (s_num_ <= 0) {
    return nullptr;
  }
  return res_->StringAttrs() + (s_cur_++ * s_num_);
}

}  // namespace op
}  // namespace graphlearn
