/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "boost/program_options.hpp"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "grpc/support/time.h"
#include "grpcpp/create_channel.h"
#include "seastar/core/print.hh"

#include "common/host.h"
#include "common/log.h"
#include "common/options.h"
#include "core/execution/dag.h"
#include "generated/fbs/install_query_req_generated.h"
#include "generated/proto/coordinator.grpc.pb.h"

namespace bpo = boost::program_options;
using namespace dgs;
using namespace dgs::execution;

/// 1. Start coordinator.py
/// 2. Call Java client to install query.

int main(int argc, char* argv[]) {
  auto &coord_option = Options::GetInstance().GetCoordClientOptions();
  auto channel = grpc::CreateChannel(coord_option.server_ipaddr,
      grpc::InsecureChannelCredentials());
  auto s = channel->WaitForConnected(gpr_time_add(gpr_now(GPR_CLOCK_REALTIME),
      gpr_time_from_seconds(coord_option.wait_time_in_sec, GPR_TIMESPAN)));
  std::unique_ptr<Coordinator::Stub> stub = Coordinator::NewStub(channel);
  grpc::Status status;

  grpc::ClientContext reg_ctx;
  RegisterWorkerRequestPb  reg_req;
  RegisterWorkerResponsePb reg_res;
  reg_req.set_worker_type(WorkerType::Serving);
  int32_t worker_id = 0;
  reg_req.set_worker_id(worker_id);
  auto ipaddr = GetLocalEndpoint(GetAvailablePort());
  reg_req.set_worker_ip(ipaddr);
  status = stub->RegisterWorker(&reg_ctx, reg_req, &reg_res);

  grpc::ClientContext init_ctx;
  GetInitInfoRequestPb  init_req;
  GetInitInfoResponsePb init_res;
  init_req.set_worker_type(WorkerType::Serving);
  init_req.set_worker_id(worker_id);
  status = stub->GetInitInfo(&init_ctx, init_req, &init_res);
  std::string query_plan_req = init_res.serving_info().query_plan();
  fmt::print("QueryPlan: {}", query_plan_req);

  std::string schemafile;
  std::string jsonfile;
  const char* default_schema = "../../fbs/install_query_req.fbs";
  bool ok;
  ok = flatbuffers::LoadFile(default_schema, false, &schemafile);
  if (!ok) { LOG(FATAL) << "Load install_query_request schema file failed.\n"; }

  flatbuffers::Parser parser;
  // parser.opts.strict_json = true;
  const char* include_paths[] = { "../../fbs/" };
  ok = parser.Parse(schemafile.c_str(), include_paths);
  if (!ok) {
    LOG(FATAL) << "Parse install_query_request schema file failed.\n";
  }
  ok = parser.Parse(query_plan_req.c_str());
  if (!ok) { LOG(FATAL) << "Parse install_query_request json file failed.\n"; }

  auto* ptr = reinterpret_cast<char*>(parser.builder_.GetBufferPointer());
  auto *rep = GetInstallQueryRequestRep(ptr);
  Dag dag(rep->query_plan());
  dag.DebugInfo();
}
