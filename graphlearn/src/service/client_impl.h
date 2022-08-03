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

#ifndef GRAPHLEARN_SERVICE_CLIENT_IMPL_H_
#define GRAPHLEARN_SERVICE_CLIENT_IMPL_H_

#include <string>
#include "include/constants.h"
#include "include/dag_request.h"
#include "include/op_request.h"
#include "include/status.h"
#include "generated/proto/request.pb.h"

namespace graphlearn {

class ClientImpl {
public:
  virtual ~ClientImpl() = default;

  virtual Status RunOp(const OpRequest* req, OpResponse* res) = 0;
  virtual Status Stop() = 0;
  virtual Status Report(const StateRequestPb* req) = 0;
  virtual Status RunDag(const DagRequest* req) = 0;
  virtual Status GetDagValues(const GetDagValuesRequest* req,
                              GetDagValuesResponse* res) = 0;
  virtual std::vector<int32_t> GetOwnServers() {
    std::vector<int32_t> empty;
    return empty;
  }

protected:
  ClientImpl() = default;
};

ClientImpl* NewInMemoryClientImpl();
ClientImpl* NewRpcClientImpl(int32_t server_id, bool server_own = true);

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_CLIENT_IMPL_H_
