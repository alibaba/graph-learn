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

#ifndef GRAPHLEARN_SERVICE_DIST_GRPC_SERVICE_H_
#define GRAPHLEARN_SERVICE_DIST_GRPC_SERVICE_H_

#include "graphlearn/proto/service.grpc.pb.h"
#include "graphlearn/proto/service.pb.h"
#include "grpcpp/grpcpp.h"

namespace graphlearn {

class Env;
class Executor;
class Coordinator;
class RequestFactory;

class GrpcServiceImpl : public GraphLearn::Service {
public:
  GrpcServiceImpl(Env* env, Executor* executor, Coordinator* coord);
  virtual ~GrpcServiceImpl();

  ::grpc::Status HandleOp(
      ::grpc::ServerContext* context,
      const OpRequestPb* request,
      OpResponsePb* response) override;

  ::grpc::Status HandleStop(
      ::grpc::ServerContext* context,
      const StopRequestPb* request,
      StopResponsePb* response) override;

  ::grpc::Status HandleReport(
      ::grpc::ServerContext* context,
      const StateRequestPb* request,
      StateResponsePb* response) override;


private:
  Env*         env_;
  Executor*    executor_;
  Coordinator* coord_;
  RequestFactory* factory_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_DIST_GRPC_SERVICE_H_
