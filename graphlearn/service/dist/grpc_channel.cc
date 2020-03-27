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

#include "graphlearn/service/dist/grpc_channel.h"

#include <unistd.h>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/threading/sync/lock.h"

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

}  // anonymous namespace

GrpcChannel::GrpcChannel(const std::string& endpoint)
    : broken_(false), endpoint_(endpoint) {
  if (endpoint.empty()) {
    broken_ = true;
  } else {
    NewChannel(endpoint);
  }
}

GrpcChannel::~GrpcChannel() {
}

void GrpcChannel::MarkBroken() {
  ScopedLocker<std::mutex> _(&mtx_);
  broken_ = true;
}

bool GrpcChannel::IsBroken() const {
  return broken_;
}

void GrpcChannel::Reset(const std::string& endpoint) {
  ScopedLocker<std::mutex> _(&mtx_);
  NewChannel(endpoint);
  broken_ = false;
  endpoint_ = endpoint;
  LOG(WARNING) << "Reset channel from " << endpoint_ << " to " << endpoint;
}

Status GrpcChannel::CallMethod(const OpRequestPb* req, OpResponsePb* res) {
  if (broken_) {
    return error::Unavailable("Channel is broken, please retry later");
  }

  ::grpc::ClientContext ctx;
  ::grpc::Status s = stub_->HandleOp(&ctx, *req, res);
  return Transmit(s);
}

Status GrpcChannel::CallStop(const StopRequestPb* req, StopResponsePb* res) {
  if (broken_) {
    return error::Unavailable("Channel is broken, please retry later");
  }

  ::grpc::ClientContext ctx;
  ::grpc::Status s = stub_->HandleStop(&ctx, *req, res);
  return Transmit(s);
}

void GrpcChannel::NewChannel(const std::string& endpoint) {
  channel_ = ::grpc::CreateChannel(
      endpoint,
      ::grpc::InsecureChannelCredentials());
  stub_ = GraphLearn::NewStub(channel_);
}

}  // namespace graphlearn
