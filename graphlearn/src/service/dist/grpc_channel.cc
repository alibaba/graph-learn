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

#include "service/dist/grpc_channel.h"

#include "common/base/errors.h"
#include "common/base/log.h"
#include "common/threading/sync/lock.h"
#include "include/config.h"

namespace graphlearn {

namespace {

Status Transmit(const ::grpc::Status& s) {
  if (s.ok()) {
    return Status::OK();
  } else {
    return Status(
      static_cast<error::Code>(static_cast<int>(s.error_code())),
      s.error_message());
  }
}

void SetContext(::grpc::ClientContext* ctx) {
  auto deadline = std::chrono::system_clock::now() +
    std::chrono::milliseconds(GLOBAL_FLAG(Timeout) * 1000);
  ctx->set_deadline(deadline);
}

}  // anonymous namespace

GrpcChannel::GrpcChannel(const std::string& endpoint)
    : endpoint_(endpoint) {
  if (endpoint.empty()) {
    broken_.store(true);
  } else {
    broken_.store(false);
    NewChannel(endpoint);
  }
  stopped_.store(false);
}

GrpcChannel::~GrpcChannel() {
}

void GrpcChannel::MarkBroken() {
  ScopedLocker<std::mutex> _(&mtx_);
  broken_.store(true);
}

bool GrpcChannel::IsBroken() const {
  return broken_.load();
}

bool GrpcChannel::IsStopped() const {
  return stopped_.load();
}

void GrpcChannel::Reset(const std::string& endpoint) {
  ScopedLocker<std::mutex> _(&mtx_);
  NewChannel(endpoint);
  broken_.store(false);
  stopped_.store(false);
  endpoint_ = endpoint;
  LOG(WARNING) << "Reset channel from " << endpoint_ << " to " << endpoint;
}

Status GrpcChannel::CallMethod(const OpRequestPb* req, OpResponsePb* res) {
  if (broken_.load()) {
    return error::Unavailable("Channel is broken, please retry later");
  }

  ::grpc::ClientContext ctx;
  SetContext(&ctx);
  ::grpc::Status s = stub_->HandleOp(&ctx, *req, res);
  return Transmit(s);
}

Status GrpcChannel::CallDag(const DagDef* req, StatusResponsePb* res) {
  if (broken_.load()) {
    return error::Unavailable("Channel is broken, please retry later");
  }

  ::grpc::ClientContext ctx;
  SetContext(&ctx);
  ::grpc::Status s = stub_->HandleDag(&ctx, *req, res);
  return Transmit(s);
}

Status GrpcChannel::CallDagValues(const DagValuesRequestPb* req,
                                  DagValuesResponsePb* res) {
  if (broken_.load()) {
    return error::Unavailable("Channel is broken, please retry later");
  }

  ::grpc::ClientContext ctx;
  SetContext(&ctx);
  ::grpc::Status s = stub_->HandleDagValues(&ctx, *req, res);
  return Transmit(s);
}

Status GrpcChannel::CallStop(const StopRequestPb* req, StatusResponsePb* res) {
  // TODO(tao): do we need such a check ?
  //
  // if (stopped_.load()) {
  //   return Status::OK();
  // }
  stopped_.store(true);
  if (broken_.load()) {
    return error::Unavailable("Channel is broken, please retry later");
  }

  ::grpc::ClientContext ctx;
  SetContext(&ctx);
  ::grpc::Status s = stub_->HandleStop(&ctx, *req, res);
  return Transmit(s);
}

Status GrpcChannel::CallReport(const StateRequestPb* req,
                               StatusResponsePb* res) {
  if (broken_.load()) {
    return error::Unavailable("Channel is broken, please retry later");
  }

  ::grpc::ClientContext ctx;
  SetContext(&ctx);
  ::grpc::Status s = stub_->HandleReport(&ctx, *req, res);
  return Transmit(s);
}

void GrpcChannel::NewChannel(const std::string& endpoint) {
  grpc::ChannelArguments args;
  args.SetMaxSendMessageSize(-1);
  args.SetMaxReceiveMessageSize(-1);
  channel_ = ::grpc::CreateCustomChannel(
      endpoint,
      ::grpc::InsecureChannelCredentials(),
      args);
  stub_ = GraphLearn::NewStub(channel_);
}

}  // namespace graphlearn
