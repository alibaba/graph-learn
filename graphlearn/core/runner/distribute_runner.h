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

#ifndef GRAPHLEARN_CORE_RUNNER_DISTRIBUTE_RUNNER_H_
#define GRAPHLEARN_CORE_RUNNER_DISTRIBUTE_RUNNER_H_

#include <memory>
#include <string>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/rpc/notification.h"
#include "graphlearn/common/threading/runner/threadpool.h"
#include "graphlearn/core/operator/operator.h"
#include "graphlearn/include/op_request.h"
#include "graphlearn/include/shardable.h"
#include "graphlearn/include/status.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {

template <class Request, class Response>
class Runner {
public:
  Runner(Env* env, op::Operator* op) : env_(env), op_(op) {}
  virtual ~Runner() = default;

  virtual Status Run(const Request* req, Response* res) {
    return op_->Process(req, res);
  }

private:
  Env* env_;
  op::Operator* op_;
};

template <class Request, class Response>
class DistributeRunner : public Runner<Request, Response> {
public:
  DistributeRunner(Env* env, int32_t local_id, op::Operator* op)
      : Runner<Request, Response>(env, op),
        env_(env),
        local_id_(local_id),
        op_(op) {
  }

  virtual ~DistributeRunner() = default;

  Status Run(const Request* req, Response* res) override {
    if (!req->IsShardable()) {
      return Runner<Request, Response>::Run(req, res);
    } else {
      ShardsPtr<Request> req_shards = req->Partition();
      ShardsPtr<Response> res_shards(
        new Shards<Response>(req_shards->Capacity()));
      ShardsPtr<Status> status_shards(
        new Shards<Status>(req_shards->Capacity()));

      RunInParallel(req->Name(), res, req_shards, res_shards, status_shards);

      int32_t shard_id = 0;
      Status* s = new Status();
      while (status_shards->Next(&shard_id, &s)) {
        if (!s->ok()) {
          return *s;
        }
      }

      res_shards->StickerPtr()->CopyFrom(*(req_shards->StickerPtr()));
      res->Stitch(res_shards);
      return *s;
    }
  }

private:
  void RunInParallel(const std::string& name,
                     Response* res,
                     ShardsPtr<Request> shards,
                     ShardsPtr<Response> ret,
                     ShardsPtr<Status> ret_status) {
    auto notifier = Init(name, shards->Size());
    ThreadPool* tp = env_->InterThreadPool();

    int32_t shard_id = 0;
    Request* shard_req = nullptr;
    while (shards->Next(&shard_id, &shard_req)) {
      notifier->AddRpcTask(shard_id);

      Response* shard_res = res->New();
      ret->Add(shard_id, shard_res, true);

      Status* s = new Status();
      ret_status->Add(shard_id, s, true);

      tp->AddTask(
        NewClosure(
          this, &DistributeRunner<Request, Response>::DoRun,
          shard_id,
          static_cast<const Request*>(shard_req),
          shard_res,
          s,
          notifier));
    }
    notifier->Wait();
  }

  std::shared_ptr<RpcNotification> Init(const std::string& name,
                                        int32_t size) {
    auto notifier = std::make_shared<RpcNotification>();
    notifier->Init(name, size);
    notifier->SetCallback([](const std::string& req_type,
                             const Status& status) {
      if (!status.ok()) {
        LOG(ERROR) << "Rpc failed:" << status.ToString()
                   << "name:" << req_type;
      }
    });
    return notifier;
  }

  void DoRun(int32_t shard_id,
             const Request* req,
             Response* res,
             Status* s,
             std::shared_ptr<RpcNotification> notifier) {
    op::RemoteOperator* op = static_cast<op::RemoteOperator*>(op_);
    if (shard_id == local_id_) {
      *s = op->Process(req, res);
    } else {
      *s = op->Call(shard_id, req, res);
    }

    if (s->ok()) {
      notifier->Notify(shard_id);
    } else {
      notifier->NotifyFail(shard_id, *s);
    }
  }

private:
  Env*          env_;
  int32_t       local_id_;
  op::Operator* op_;
};

typedef Runner<OpRequest, OpResponse> OpRunner;
typedef DistributeRunner<OpRequest, OpResponse> DistOpRunner;

std::unique_ptr<OpRunner> GetOpRunner(Env* env, op::Operator* op);

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_RUNNER_DISTRIBUTE_RUNNER_H_
