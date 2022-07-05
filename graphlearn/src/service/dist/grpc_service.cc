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

#include "service/dist/grpc_service.h"

#include "common/base/errors.h"
#include "common/base/log.h"
#include "include/op_request.h"
#include "platform/env.h"
#include "service/dist/coordinator.h"
#include "service/executor.h"

namespace graphlearn {

namespace {

::grpc::Status Transmit(const Status& s) {
  if (s.ok()) {
    return ::grpc::Status::OK;
  } else {
    return ::grpc::Status(
      static_cast<::grpc::StatusCode>(static_cast<int>(s.code())),
      s.msg());
  }
}

}  // anonymous namespace

GrpcServiceImpl::GrpcServiceImpl(Env* env, Executor* executor,
                                 Coordinator* coord)
    : env_(env), executor_(executor), coord_(coord) {
  factory_ = RequestFactory::GetInstance();
}

GrpcServiceImpl::~GrpcServiceImpl() {
}

::grpc::Status GrpcServiceImpl::HandleOp(
    ::grpc::ServerContext* context,
    const OpRequestPb* request,
    OpResponsePb* response) {
  if (!request->need_server_ready()) {
    // Just go ahead
  } else if (!coord_->IsReady()) {
    Status s = error::Unavailable("Not all servers ready, please retry later");
    return Transmit(s);
  }
  if (context->IsCancelled()) {
    Status s = error::DeadlineExceeded("Deadline exceeded or client cancelled");
    return Transmit(s);
  }

  OpRequestPtr req(factory_->NewRequest(request->name()));
  OpResponsePtr res(factory_->NewResponse(request->name()));
  req->ParseFrom(request);
  Status s = executor_->RunOp(req.get(), res.get());
  if (s.ok()) {
    res->SerializeTo(response);
  }
  return Transmit(s);
}

::grpc::Status GrpcServiceImpl::HandleStop(
    ::grpc::ServerContext* context,
    const StopRequestPb* request,
    StatusResponsePb* response) {
  Status s = coord_->Stop(request->client_id(), request->client_count());
  return Transmit(s);
}

::grpc::Status GrpcServiceImpl::HandleReport(
    ::grpc::ServerContext* context,
    const StateRequestPb* request,
    StatusResponsePb* response) {
  SystemState state = static_cast<SystemState>(request->state());
  Status s;
  switch (state) {
  case kStarted:
    s = coord_->SetStarted(request->id());
    break;
  case kInited:
    s = coord_->SetInited(request->id());
    break;
  case kReady:
    s = coord_->SetReady(request->id());
    break;
  case kStopped:
    s = coord_->SetStopped(request->id(), request->count());
    break;
  default:
    LOG(INFO) << "Set reserved state: " << state;
    s = coord_->SetState(request->state(), request->id());
    break;
  }
  return Transmit(s);
}

::grpc::Status GrpcServiceImpl::HandleDag(
    ::grpc::ServerContext* context,
    const DagDef* request,
    StatusResponsePb* response) {
  if (!coord_->IsReady()) {
    Status s = error::Unavailable("Not all servers ready, please retry later");
    return Transmit(s);
  }

  Status s = executor_->RunDag(*request);
  return Transmit(s);
}

::grpc::Status GrpcServiceImpl::HandleDagValues(
    ::grpc::ServerContext* context,
    const DagValuesRequestPb* request,
    DagValuesResponsePb* response) {
  if (!coord_->IsReady()) {
    Status s = error::Unavailable("Not all servers ready, please retry later");
    return Transmit(s);
  }

  GetDagValuesRequest req(request->id(), request->client_id());
  GetDagValuesResponse res;
  Status s = executor_->GetDagValues(&req, &res);
  if (s.ok()) {
    res.SerializeTo(response);
  }
  return Transmit(s);
}

}  // namespace graphlearn
