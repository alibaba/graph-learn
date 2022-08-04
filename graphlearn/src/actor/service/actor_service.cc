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

#include "actor/service/actor_service.h"

#include <semaphore.h>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "brane/actor/actor-system.hh"
#include "brane/core/root_actor_group.hh"
#include "brane/core/shard-config.hh"
#include "brane/core/coordinator.hh"
#include "actor/graph/loader_config.h"
#include "actor/graph/sharded_graph_store.h"
#include "actor/service/actor_coord.h"
#include "common/base/host.h"
#include "common/base/macros.h"
#include "include/config.h"
#include "platform/env.h"
#include "seastar/core/alien.hh"

namespace graphlearn {
namespace act {

static sem_t ActorIsReadyFlag;

void LaunchActorSystem(int32_t server_id,
                       int32_t server_count,
                       bool distributed_mode) {
  char prog_name[] = "graph_store_app";
  char docker_opt[] = "--thread-affinity=0";
  char cores[16], mach_id[16], server_list[32];
  snprintf(cores, sizeof(cores), "-c%d", GLOBAL_FLAG(LocalShardCount));
  snprintf(mach_id, sizeof(mach_id), "--machine-id=%d", server_id);
  snprintf(server_list, sizeof(server_list),
    "--worker-node-list=s-%d.list", server_id);

  // FIXME: find the best value.
  unsigned best_p2p_conn_count = GLOBAL_FLAG(LocalShardCount) / 2;
  char p2p_conn_count[64];
  snprintf(p2p_conn_count, sizeof(p2p_conn_count),
    "--p2p-connection-count=%d", best_p2p_conn_count);

  int argc = distributed_mode ? 6 : 3;
  char* argv[] = {prog_name, cores, docker_opt,
    server_list, mach_id, p2p_conn_count};

  seastar::app_template::config conf;
  conf.auto_handle_sigint_sigterm = false;
  brane::actor_system sys{std::move(conf)};
  sys.run(argc, argv, [] {
    sem_post(&ActorIsReadyFlag);
    return seastar::make_ready_future<>();
  });
}

ActorService::ActorService(int32_t server_id,
                           int32_t server_count,
                           Coordinator* coord)
    : server_id_(server_id),
      server_count_(server_count),
      coord_(coord) {
  sem_init(&ActorIsReadyFlag, 0, 0);
}

ActorService::~ActorService() {
  sem_destroy(&ActorIsReadyFlag);
}

Status ActorService::StartActorSystem(bool distributed_mode) {
  actor_system_ = std::thread(LaunchActorSystem,
                              server_id_, server_count_,
                              distributed_mode);
  sem_wait(&ActorIsReadyFlag);
  return Status::OK();
}

void ActorService::WriteServerConfigToTmpFile(
    std::unique_ptr<NamingEngine>&& engine) {
  char list_fname[16];
  snprintf(list_fname, sizeof(list_fname), "s-%d.list", server_id_);
  std::ofstream tmp_file(list_fname);
  tmp_file << "mach_id, #cores, ip_addr" << std::endl;
  for (int32_t sid = 0; sid < server_count_; ++sid) {
    tmp_file << sid << " " << engine->Get(sid) << std::endl;
  }
}

Status ActorService::Start() {
  Status s;
  bool distributed_mode = GLOBAL_FLAG(DeployMode) != kLocal;

  if (distributed_mode) {
    /// If sync with rpc, ActorCoordImpl use RpcCoordinator to sync state.
    /// Else, when sync with file system, record sync message for actor.
    brane::coordinator::get().set_impl(
      std::unique_ptr<brane::coordinator::impl>(
      new ActorCoordImpl(coord_, server_id_, server_count_)));

    if (GLOBAL_FLAG(TrackerMode) == kFileSystem) {
      std::unique_ptr<NamingEngine> engine(new FSNamingEngine("actor"));
      engine->SetCapacity(server_count_);

      std::string content = std::to_string(GLOBAL_FLAG(LocalShardCount))
          + " " + GetLocalEndpoint(GetAvailablePort());

      s = engine->Update(server_id_, content);
      LOG_RETURN_IF_NOT_OK(s)
      while (engine->Size() < server_count_) {
        sleep(1);
      }
      WriteServerConfigToTmpFile(std::move(engine));
    }
  }
  return StartActorSystem(distributed_mode);
}

Status ActorService::Init() {
  if (GLOBAL_FLAG(DeployMode) != kLocal) {
    Status s = coord_->Init();
    RETURN_IF_NOT_OK(s)

    while (!coord_->IsInited()) {
      sleep(1);
    }
    return s;
  }
  return Status::OK();
}

Status ActorService::Build() {
  if (GLOBAL_FLAG(DeployMode) != kLocal) {
    Status s = coord_->Prepare();
    RETURN_IF_NOT_OK(s)

    while (!coord_->IsReady()) {
      sleep(1);
    }
    return s;
  }
  return Status::OK();
}

Status ActorService::Stop() {
  if (GLOBAL_FLAG(DeployMode) != kLocal) {
    while (!coord_->IsStopped()) {
      LOG(WARNING) << "Waiting other servers to stop";
      sleep(1);
    }
  }

  Env::Default()->SetStopping();

  if (server_id_ == 0) {
    seastar::alien::run_on(0, [] {
      brane::actor_engine().exit(true);
    });
  }
  actor_system_.join();

  actor::ShardedGraphStore::Get().Finalize();
  return Status::OK();
}

}  // namespace act
}  // namespace graphlearn
