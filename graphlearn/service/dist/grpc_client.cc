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

#include <unistd.h>
#include <memory>
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/base/errors.h"
#include "graphlearn/include/config.h"
#include "graphlearn/service/client_impl.h"
#include "graphlearn/service/dist/channel_manager.h"
#include "graphlearn/service/dist/grpc_channel.h"

namespace graphlearn {

class GrpcClientImpl : public ClientImpl {
public:
  GrpcClientImpl(int32_t server_id, bool server_own) : server_own_(server_own) {
    if (!server_own_) {
      InitGoogleLogging();
    }

    manager_ = ChannelManager::GetInstance();
    manager_->SetCapacity(GLOBAL_FLAG(ServerCount));

    if (server_id == -1) {
      channel_ = manager_->AutoSelect();
    } else {
      channel_ = manager_->ConnectTo(server_id);
    }
  }

  virtual ~GrpcClientImpl() {
    if (!server_own_) {
      UninitGoogleLogging();
    }
  }

  Status RunOp(const OpRequest* request,
               OpResponse* response) override {
    std::unique_ptr<OpRequestPb> req(new OpRequestPb);
    std::unique_ptr<OpResponsePb> res(new OpResponsePb);
    const_cast<OpRequest*>(request)->SerializeTo(req.get());

    Status s = channel_->CallMethod(req.get(), res.get());
    int32_t retry = 1;
    while (IsRetryable(s) && retry < GLOBAL_FLAG(RetryTimes)) {
      channel_->MarkBroken();
      sleep(1 << retry);
      s = channel_->CallMethod(req.get(), res.get());
      ++retry;
    }
    if (s.ok()) {
      response->ParseFrom(res.get());
    }
    return s;
  }

  Status Stop() override {
    StopRequestPb req;
    req.set_client_id(GLOBAL_FLAG(ClientId));
    req.set_client_count(GLOBAL_FLAG(ClientCount));
    StatusResponsePb res;

    Status s = channel_->CallStop(&req, &res);
    int32_t retry = 1;
    while (IsRetryable(s) && retry < GLOBAL_FLAG(RetryTimes)) {
      channel_->MarkBroken();
      sleep(1 << retry);
      s = channel_->CallStop(&req, &res);
      ++retry;
    }
    manager_->Stop();
    return Status::OK();
  }

  Status Report(const StateRequestPb* req) override {
    StatusResponsePb res;
    Status s = channel_->CallReport(req, &res);
    int32_t retry = 1;
    while (IsRetryable(s) && retry < GLOBAL_FLAG(RetryTimes)) {
      channel_->MarkBroken();
      sleep(1 << retry);
      s = channel_->CallReport(req, &res);
      ++retry;
    }
    return Status::OK();
  }

  Status RunDag(const DagRequest* request) override {
    StatusResponsePb res;
    Status s = channel_->CallDag(&(request->def_), &res);
    int32_t retry = 1;
    while (IsRetryable(s) && retry < GLOBAL_FLAG(RetryTimes)) {
      channel_->MarkBroken();
      sleep(1 << retry);
      s = channel_->CallDag(&(request->def_), &res);
      ++retry;
    }
    return s;
  }

  Status GetDagValues(const GetDagValuesRequest* request,
                      GetDagValuesResponse* response) override {
    std::unique_ptr<DagValuesRequestPb> req(new DagValuesRequestPb);
    std::unique_ptr<DagValuesResponsePb> res(new DagValuesResponsePb);
    const_cast<GetDagValuesRequest*>(request)->SerializeTo(req.get());

    Status s = channel_->CallDagValues(req.get(), res.get());
    int32_t retry = 1;
    while (IsRetryable(s) && retry < GLOBAL_FLAG(RetryTimes)) {
      channel_->MarkBroken();
      sleep(1 << retry);
      s = channel_->CallDagValues(req.get(), res.get());
      ++retry;
    }
    if (s.ok()) {
      response->ParseFrom(res.get());
    }
    return s;
  }

  std::vector<int32_t> GetOwnServers() override {
    return manager_->GetOwnServers();
  }

private:
  bool IsRetryable(const Status& s) {
    return error::IsUnavailable(s) || error::IsDeadlineExceeded(s);
  }

private:
  ChannelManager* manager_;
  GrpcChannel*    channel_;
  bool            server_own_;
};

ClientImpl* NewRpcClientImpl(int32_t server_id, bool server_own) {
  return new GrpcClientImpl(server_id, server_own);
}

}  // namespace graphlearn
