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

#ifndef GRAPHLEARN_SERVICE_DIST_GRPC_CHANNEL_H_
#define GRAPHLEARN_SERVICE_DIST_GRPC_CHANNEL_H_

#include <cstdint>
#include <memory>
#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include "graphlearn/include/status.h"
#include "graphlearn/proto/service.grpc.pb.h"
#include "graphlearn/proto/service.pb.h"
#include "grpcpp/grpcpp.h"

namespace graphlearn {

class GrpcChannel {
public:
  explicit GrpcChannel(const std::string& endpoint);
  ~GrpcChannel();

  void MarkBroken();
  bool IsBroken() const;
  void Reset(const std::string& endpoint);

  Status CallMethod(const OpRequestPb* req, OpResponsePb* res);
  Status CallStop(const StopRequestPb* req, StopResponsePb* res);

private:
  void NewChannel(const std::string& endpoint);

private:
  std::mutex mtx_;
  volatile bool broken_;
  std::string endpoint_;
  std::shared_ptr<::grpc::Channel> channel_;
  std::unique_ptr<GraphLearn::Stub> stub_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_DIST_GRPC_CHANNEL_H_
