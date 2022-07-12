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

#ifndef DGS_SERVICE_CHANNEL_RECORD_POLLER_H_
#define DGS_SERVICE_CHANNEL_RECORD_POLLER_H_

#include <semaphore.h>

#include <atomic>
#include <deque>
#include <future>
#include <thread>
#include <unordered_map>

#include "cppkafka/consumer.h"

#include "common/options.h"
#include "common/partition_router.h"
#include "common/partitioner.h"
#include "service/actor_ref_builder.h"

namespace dgs {

class RecordBatchIngestor {
public:
  RecordBatchIngestor(PartitionRouter* router, uint32_t poller_id);
  std::future<size_t> operator()(actor::BytesBuffer&& buf);

private:
  std::vector<SamplingActor_ref> actor_refs_;
  PartitionRouter*               partition_router_;
  const uint32_t                 poller_id_;
  const uint32_t                 gsid_anchor_;
};

class SampleBatchIngestor {
public:
  SampleBatchIngestor(PartitionRouter* router,
                      Partitioner* partitioner,
                      uint32_t poller_id);
  std::future<size_t> operator()(actor::BytesBuffer&& buf);

private:
  std::vector<DataUpdateActor_ref> actor_refs_;
  PartitionRouter*                 partition_router_;
  Partitioner*                     partitioner_;
  const uint32_t                   poller_id_;
  const uint32_t                   gsid_anchor_;
  uint32_t                         shard_idx_ = 0;
  std::vector<std::vector<storage::KVPair>> partition_records_;
};

/// The basic io records polling unit, which corresponds to a
/// specific sample builder partition.
class RecordPoller {
public:
  RecordPoller(uint32_t poller_id,
               WorkerType worker_type,
               const RecordPollingOptions& opts,
               uint32_t kafka_partition_id,
               int64_t ingested_offset,
               PartitionRouter* router,
               Partitioner* partitioner);
  ~RecordPoller();

  /// Try to poll an io message without timeout and send the records
  /// to related actor for ingesting.
  ///
  /// Calling this method will first check all the pending futures
  /// of previous record ingesting tasks and try to finish them.
  ///
  /// \returns @true if there is one io message polled; and @false
  /// if the current concurrency reached the maximum limit, or no
  /// io message has been polled.
  bool PollAndIngest();

  int64_t GetIngestedOffset() const {
    return ingested_offset_.load(std::memory_order_relaxed);
  }

  uint32_t GetMaxConcurrency() const {
    return max_concurrency_.load();
  }

  void SetMaxConcurrency(uint32_t value) {
    max_concurrency_.store(value);
    concur_mutated_.store(true);
  }

private:
  void CheckPendingFutures();
  void IngestBatch(cppkafka::Message&& msg);

  static bool IsReady(const std::future<size_t>& fut);

private:
  struct IngestFuture {
    int64_t offset;
    std::future<size_t> fut;
    bool concurrency_released = false;
    IngestFuture(int64_t offset, std::future<size_t>&& fut)
      : offset(offset), fut(std::move(fut)) {}
    IngestFuture(const IngestFuture&) = delete;
    IngestFuture& operator=(const IngestFuture&) = delete;
    IngestFuture(IngestFuture&&) = default;
    IngestFuture& operator=(IngestFuture&&) = default;
  };

private:
  const uint32_t        poller_id_;
  // kafka partition id to consume.
  const uint32_t        kafka_partition_id_;
  uint32_t              concurrency_;
  std::atomic<uint32_t> max_concurrency_;
  std::atomic<bool>     concur_mutated_{false};
  // -1 mean no ingested data.
  std::atomic<int64_t> ingested_offset_;
  std::function<std::future<size_t>(actor::BytesBuffer&&)> ingestor_;
  std::unique_ptr<cppkafka::Consumer>                      consumer_;
  std::deque<IngestFuture> pending_futures_;

#ifdef DGS_BENCHMARK

public:
  uint64_t num_processed_records() const {
    return num_processed_records_.load(std::memory_order_relaxed);
  }

  uint64_t num_processed_bytes() const {
    return num_processed_bytes_.load(std::memory_order_relaxed);
  }

private:
  std::atomic<uint64_t> num_processed_records_{0};
  std::atomic<uint64_t> num_processed_bytes_{0};
#endif
};

/// The record polling thread wrapper.
/// One record polling worker may correspond to multiple record pollers.
class RecordPollingWorker {
public:
  RecordPollingWorker(const std::string& worker_name,
                      uint32_t retry_interval_ms,
                      std::vector<RecordPoller*>&& pollers);

  ~RecordPollingWorker();

  /// Start or stop the io polling for each poller in this worker.
  void Start();
  void Stop();

private:
  void PollForEach();

private:
  const std::string            worker_name_;
  std::vector<RecordPoller*>   record_pollers_;
  std::chrono::milliseconds    retry_interval_;
  std::atomic<bool>            stopped_ = { true };
  std::unique_ptr<std::thread> worker_thread_;
};

/// Manage all record polling workers by machine.
class RecordPollingManager {
public:
  RecordPollingManager() = default;
  ~RecordPollingManager() = default;

  void UpdatePartitionRoutingInfo(const std::vector<RoutingUpdate>& updates) {
      partition_router_.UpdatePartitionRoutingInfo(updates);
  }

  void Init(Partitioner&& partitioner,
            const std::vector<ShardId>& store_partition_routing_info,
            const std::vector<uint32_t>& kafka_partitions,
            const std::unordered_map<uint32_t, int64_t>& ready_offsets);
  void Start();
  void Stop();

  void BlockUntilReady();

  void SetConcurrency(uint32_t value);
  void GetConcurrency();

  std::unordered_map<uint32_t, int64_t> GetIngestedOffsets();

private:
  using RecordPollerPtr = std::unique_ptr<RecordPoller>;
  using PollingWorkerPtr = std::unique_ptr<RecordPollingWorker>;

  Partitioner                                       partitioner_;
  PartitionRouter                                   partition_router_;
  std::vector<std::pair<uint32_t, RecordPollerPtr>> kafka_pid_to_pollers_;
  std::vector<PollingWorkerPtr>                     workers_;

#ifdef DGS_BENCHMARK
  uint64_t bulkload_start_time_ = 0;
  uint64_t bulkload_end_time_ = 0;
#endif
};

}  // namespace dgs

#endif  // DGS_SERVICE_CHANNEL_RECORD_POLLER_H_
