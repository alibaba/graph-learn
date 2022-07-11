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

#ifndef DGS_SERVICE_ADAPTIVE_RATE_LIMITER_H_
#define DGS_SERVICE_ADAPTIVE_RATE_LIMITER_H_

#include <atomic>
#include <thread>

#include "boost/lockfree/spsc_queue.hpp"

#include "common/actor_wrapper.h"

namespace dgs {

class RecordPollingManager;

class AdaptiveRateLimiter {
  const uint64_t k_p99_latency_ms_ = 18;
  const uint32_t k_interval_sec_;
  const uint32_t k_window_size_ = 10;
  const uint32_t k_max_lat_size_ = 10000;

public:
  explicit AdaptiveRateLimiter(RecordPollingManager* kp_manager,
                               uint32_t check_interval_sec = 10);
  ~AdaptiveRateLimiter();

  void Start();
  void Stop();

  bool SubmitLatency(uint32_t shard_id, uint64_t latency) {
    return lat_pendings_.v.qs[shard_id].push(latency);
  }

private:
  void RunUntilTerminate();

private:
  using lf_queue = boost::lockfree::spsc_queue<uint64_t,
    boost::lockfree::capacity<1024>>;

  union sharded_lfq {
    sharded_lfq() {}
    ~sharded_lfq() {}
    void init(uint32_t size) { new (&v) vv(size); }
    void destroy() { v.~vv(); }

    struct vv {
      explicit vv(uint32_t size) : qs(size) {}
      std::vector<lf_queue> qs;
    } v;
  };

private:
  actor::CircularBuffer<uint64_t> latencies_;
  actor::CircularBuffer<uint32_t> anchors_;
  sharded_lfq                     lat_pendings_;
  std::unique_ptr<std::thread>    worker_thread_;
  std::atomic<bool>               stopped_ = { true };
  const uint32_t                  poll_kakfa_max_concur_;
  uint64_t                        num_lte_p99_lats_{0};
  uint32_t                        num_stable_p99_{0};
  uint32_t                        kafka_polling_concur_;
  RecordPollingManager*           kafka_poller_manager_;
};

}  // namespace dgs

#endif  // DGS_SERVICE_ADAPTIVE_RATE_LIMITER_H_
