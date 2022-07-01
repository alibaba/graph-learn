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

#include "graphlearn/contrib/knn/knn_op.h"

#include <memory>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/contrib/knn/index.h"
#include "graphlearn/contrib/knn/index_manager.h"
#include "graphlearn/contrib/knn/knn_request.h"
#include "graphlearn/include/client.h"

namespace graphlearn {
namespace op {

Status KnnOperator::Process(const OpRequest* req, OpResponse* res) {
  const KnnRequest* request = static_cast<const KnnRequest*>(req);
  KnnResponse* response = static_cast<KnnResponse*>(res);

  KnnIndex* index = KnnIndexManager::Instance()->Get(request->Type());
  if (index == nullptr) {
    USER_LOG("Invalid node type for KNN search.");
    LOG(ERROR) << "Not found node type for KNN index: " << request->Type();
    return error::InvalidArgument("Invalid node type.");
  }

  int32_t n = request->BatchSize();
  int32_t k = request->K();
  response->Init(n, k);

  const float* inputs = request->Inputs();
  int64_t* ids = const_cast<int64_t*>(response->Ids());
  float* distances = const_cast<float*>(response->Distances());

  index->Search(n, inputs, k, ids, distances);
  return Status();
}

Status KnnOperator::Call(int32_t remote_id,
                         const OpRequest* req,
                         OpResponse* res) {
  std::unique_ptr<Client> client(NewRpcClient(remote_id));
  return client->RunOp(req, res);
}

REGISTER_OPERATOR("KnnOperator", KnnOperator);

}  // namespace op
}  // namespace graphlearn
