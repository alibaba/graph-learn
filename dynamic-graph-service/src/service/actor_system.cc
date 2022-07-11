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

#include "service/actor_system.h"

#include <semaphore.h>

#include "hiactor/core/actor-app.hh"
#include "seastar/core/alien.hh"

#include "common/options.h"
#include "common/log.h"

namespace dgs {

static sem_t actor_system_ready;
static std::atomic<bool> actor_system_exist{false};

void ActorSystem::LaunchWorker(ActorSystem* self) {
  char prog_name[] = "actor_system";
  char docker_opt[] = "--thread-affinity=0";
  char enable_tp[] = "--open-thread-resource-pool=false";

  char num_shards[16], mach_id[16], server_list[32], p2p_conn_count[64];

  snprintf(num_shards, sizeof(num_shards), "-c%d", self->num_local_shards_);
  snprintf(mach_id, sizeof(mach_id), "--machine-id=%d", self->worker_id_);
  snprintf(server_list, sizeof(server_list),
      "--worker-node-list=s-%d.list", self->worker_id_);
  auto best_p2p_conn_val = self->num_local_shards_ / 2;
  snprintf(p2p_conn_count, sizeof(p2p_conn_count),
    "--p2p-connection-count=%d", best_p2p_conn_val);

  char* argv[] = {prog_name, docker_opt, num_shards, enable_tp,
      server_list, mach_id, p2p_conn_count};

  bool distributed_mode = (self->worker_type_ == WorkerType::Sampling)
      && (self->num_workers_ > 1);

  int argc = distributed_mode ? 7 : 4;

  if (distributed_mode) {
    LOG(INFO) << "Start actor system in distributed mode: "
              << ", worker id is " << self->worker_id_
              << "num_local_shards is " << self->num_local_shards_
              << "p2p_conn_count is " << best_p2p_conn_val;
  } else {
    LOG(INFO) << "Start actor system in standalone mode: "
              << "num_local_shards is " << self->num_local_shards_;
  }

  seastar::app_template::config conf;
  conf.auto_handle_sigint_sigterm = false;
  hiactor::actor_app sys{std::move(conf)};
  sys.run(argc, argv, [] {
    sem_post(&actor_system_ready);
    return seastar::make_ready_future<>();
  });
}

ActorSystem::ActorSystem(WorkerType worker_type,
                         uint32_t worker_id,
                         uint32_t num_workers,
                         uint32_t num_local_shards)
  : worker_type_(worker_type),
    worker_id_(worker_id),
    num_workers_(num_workers),
    num_local_shards_(num_local_shards),
    main_thread_(nullptr) {
  if (actor_system_exist.load()) {
    LOG(FATAL) << "Actor System is running. Trying to start"
               << "a new one is not allowed";
  }

  sem_init(&actor_system_ready, 0, 0);
  main_thread_ = std::make_unique<std::thread>(LaunchWorker, this);
  sem_wait(&actor_system_ready);
  sem_destroy(&actor_system_ready);
  actor_system_exist.store(true);
}

ActorSystem::~ActorSystem() {
  if (main_thread_) {
    seastar::alien::run_on(0, [] {
      hiactor::actor_engine().exit();
    });
    main_thread_->join();
  }
}

}  // namespace dgs
