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

#include "service/adaptive_rate_limiter.h"

#include "common/log.h"
#include "common/options.h"
#include "service/channel/record_poller.h"

namespace dgs {

AdaptiveRateLimiter::AdaptiveRateLimiter(RecordPollingManager* kp_manager,
                                         uint32_t check_interval_sec)
  : k_interval_sec_(check_interval_sec),
    poll_kakfa_max_concur_(
      Options::GetInstance().GetRecordPollingOptions().process_concurrency),
    kafka_poller_manager_(kp_manager) {
  kafka_polling_concur_ = poll_kakfa_max_concur_;
  LOG(INFO) << "initial kafka_polling_concur_ is " << kafka_polling_concur_;
}

AdaptiveRateLimiter::~AdaptiveRateLimiter() {}

void AdaptiveRateLimiter::Start() {
  lat_pendings_.init(act::LocalShardCount());

  stopped_.store(false, std::memory_order_relaxed);
  worker_thread_ = std::make_unique<std::thread>(
      &AdaptiveRateLimiter::RunUntilTerminate, this);
}

void AdaptiveRateLimiter::Stop() {
  stopped_.store(true, std::memory_order_relaxed);
  if (worker_thread_) {
    worker_thread_->join();
  }
  lat_pendings_.destroy();
}

void AdaptiveRateLimiter::RunUntilTerminate() {
  LOG(INFO) << "AdaptiveRateLimiter is running";
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(k_interval_sec_));
    if (stopped_.load(std::memory_order_relaxed)) {
      break;
    }

    for (auto& q : lat_pendings_.v.qs) {
      q.consume_all([this] (uint64_t latency) {
        latencies_.push_back(latency);
        if (latency <= k_p99_latency_ms_) {
          ++num_lte_p99_lats_;
        }
      });
    }
    LOG(INFO) << "#lte_p99_lats_ is " << num_lte_p99_lats_
              << ", #total latencies: " << latencies_.size();
    if (num_lte_p99_lats_ < 0.99 * latencies_.size()) {
      // decrease kafka polling concurrency.
      // FIXME(@goldenleaves): we need to fine-tune a optimial scale value.
      kafka_polling_concur_ = std::max(uint32_t(kafka_polling_concur_ / 3), 1u);
      kafka_poller_manager_->SetConcurrency(kafka_polling_concur_);
      num_stable_p99_ = 0;
    } else {
      // increase kafka polling concurrency in a conservative way
      if (++num_stable_p99_ % k_window_size_ == 0) {
        // FIXME(@goldenleaves): we need to fine-tune a optimial scale value.
        kafka_polling_concur_ = std::min(uint32_t(kafka_polling_concur_ * 1.3),
            poll_kakfa_max_concur_);
        kafka_poller_manager_->SetConcurrency(kafka_polling_concur_);
      }
    }
  }
  LOG(INFO) << "AdaptiveRateLimiter is stopped";
}

}  // namespace dgs
