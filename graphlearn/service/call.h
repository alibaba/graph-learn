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

#ifndef GRAPHLEARN_SERVICE_CALL_H_
#define GRAPHLEARN_SERVICE_CALL_H_

#include <chrono>  // NOLINT [build/c++11]
#include <future>  // NOLINT [build/c++11]
#include <utility>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/include/request.h"

namespace graphlearn {

enum MethodType {
  kUserDefinedOp = 0,
  kStop = 1,
  kOtherToExtend = 2,
};

struct StatusWrapper {
  Status             s_;
  std::promise<void> p_;

  void Wait(int32_t timeout_ms = -1) {
    auto future = p_.get_future();
    if (timeout_ms == -1) {
      future.wait();
    } else {
      auto status = future.wait_for(std::chrono::milliseconds(timeout_ms));
      if (status == std::future_status::timeout) {
        s_ = error::Cancelled("task timeout.");
      }
    }
  }

  void Signal(Status status = Status::OK()) {
    s_ = status;
    p_.set_value();
  }
};

struct Call {
  uint16_t           method_;
  const BaseRequest* req_;
  BaseResponse*      res_;
  StatusWrapper*     status_;

  Call() : method_(-1),
           req_(nullptr),
           res_(nullptr),
           status_(nullptr) {
  }

  Call(uint16_t method_id,
       const BaseRequest* req,
       BaseResponse* res,
       StatusWrapper* s)
    : method_(method_id), req_(req), res_(res), status_(s) {
  }

  Call(const Call& r) {
    method_ = r.method_;
    req_ = r.req_;
    res_ = r.res_;
    status_ = r.status_;
  }
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_CALL_H_
