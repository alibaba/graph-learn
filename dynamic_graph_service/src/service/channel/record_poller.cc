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

#include "service/channel/record_poller.h"

#include <algorithm>
#include <cmath>

#ifdef DGS_BENCHMARK
#include <iomanip>
#endif

#include "seastar/core/alien.hh"
#include "seastar/core/when_all.hh"

#include "common/log.h"
#include "common/utils.h"
#include "core/io/record.h"
#include "service/serving_group.actg.h"

namespace dgs {

RecordBatchIngestor::RecordBatchIngestor(PartitionRouter* router,
                                         uint32_t poller_id)
    : partition_router_(router),
      poller_id_(poller_id),
      gsid_anchor_(actor::GlobalShardIdAnchor()) {
  hiactor::scope_builder builder(0);
  actor_refs_.reserve(actor::LocalShardCount());
  for (unsigned l_sid = 0; l_sid < actor::LocalShardCount(); ++l_sid) {
    auto g_sid = gsid_anchor_ + l_sid;
    builder.set_shard(g_sid);
    actor_refs_.emplace_back(MakeSamplingActorInstRef(builder));
  }
}

std::future<size_t> RecordBatchIngestor::operator()(actor::BytesBuffer&& buf) {
  auto batch = io::RecordBatch{std::move(buf)};
  auto batch_view = batch.GetView();
  auto num_records = batch_view.RecordNum();

  PartitionId pid = batch_view.GetStorePartitionId();
  unsigned dst_l_sid = partition_router_->GetGlobalShardId(pid) - gsid_anchor_;
  return seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, dst_l_sid,
      [this, dst_l_sid, batch = std::move(batch), num_records] () mutable {
    return actor_refs_[dst_l_sid].ApplyGraphUpdates(
        std::move(batch)).then([num_records] (auto) {
      return seastar::make_ready_future<size_t>(num_records);
    });
  });
}

SampleBatchIngestor::SampleBatchIngestor(PartitionRouter* router,
                                         Partitioner* partitioner,
                                         uint32_t poller_id)
    : partition_router_(router),
      partitioner_(partitioner),
      partition_num_(partitioner->GetPartitionsNum()),
      poller_id_(poller_id),
      gsid_anchor_(actor::GlobalShardIdAnchor()) {
  hiactor::scope_builder builder(0, MakeServingGroupScope());
  actor_refs_.reserve(actor::LocalShardCount());
  for (unsigned l_sid = 0; l_sid < actor::LocalShardCount(); ++l_sid) {
    auto g_sid = gsid_anchor_ + l_sid;
    builder.set_shard(g_sid);
    actor_refs_.emplace_back(MakeDataUpdateActorInstRef(builder));
  }
}

std::future<size_t> SampleBatchIngestor::operator()(actor::BytesBuffer&& buf) {
  std::vector<storage::KVPair> partition_records[partition_num_];
  auto updates = io::SampleUpdateBatch::Deserialize(std::move(buf));
  for (auto& update : updates) {
    auto pid = partitioner_->GetPartitionId(update.key.pkey.vid);
    partition_records[pid].emplace_back(std::move(update));
  }
  std::vector<io::SampleUpdateBatch> sample_batches;
  size_t num_records = 0;
  for (PartitionId pid = 0; pid < partition_num_; ++pid) {
    if (!partition_records[pid].empty()) {
      num_records += partition_records[pid].size();
      sample_batches.emplace_back(pid, std::move(partition_records[pid]));
    }
  }
  return seastar::alien::submit_to(
      *seastar::alien::internal::default_instance,
      shard_idx_,
      [this, batches = std::move(sample_batches), num_records] () mutable {
    shard_idx_ = (shard_idx_ + 1) % actor::LocalShardCount();
    std::vector<seastar::future<>> futs;
    futs.reserve(batches.size());
    for (auto& batch : batches) {
      auto pid = batch.GetStorePartitionId();
      auto l_sid = partition_router_->GetGlobalShardId(pid) - gsid_anchor_;
      futs.emplace_back(actor_refs_[l_sid].Update(
          std::move(batch)).discard_result());
    }
    return seastar::when_all(futs.begin(), futs.end()).then(
        [num_records] (auto) {
      return seastar::make_ready_future<size_t>(num_records);
    });
  });
}

RecordPoller::RecordPoller(uint32_t poller_id,
                           WorkerType worker_type,
                           const RecordPollingOptions& opts,
                           uint32_t kafka_partition_id,
                           int64_t ingested_offset,
                           PartitionRouter* router,
                           Partitioner* partitioner)
  : poller_id_(poller_id),
    kafka_partition_id_(kafka_partition_id),
    ingested_offset_(ingested_offset),
    concurrency_(opts.process_concurrency),
    max_concurrency_(opts.process_concurrency) {
  switch (worker_type) {
    case WorkerType::Sampling: {
      ingestor_ = RecordBatchIngestor(router, poller_id_);
      break;
    }
    case WorkerType::Serving: {
      ingestor_ = SampleBatchIngestor(router, partitioner, poller_id_);
      break;
    }
    default: {
      LOG(FATAL) << "Unknown worker type! The worke_type "
                 << "should be either Sampling or Serving";
    }
  }

  consumer_ = std::make_unique<cppkafka::Consumer>(cppkafka::Configuration{
    {"metadata.broker.list", opts.FormatKafkaServers()},
    {"broker.address.family", "v4"},
    {"group.id", "record_pollers"},
    {"enable.auto.commit", false}});
  consumer_->assign({cppkafka::TopicPartition{
    opts.kafka_topic, static_cast<int32_t>(kafka_partition_id_),
    ingested_offset + 1}});
  consumer_->set_timeout(std::chrono::milliseconds(0));

  LOG(INFO) << "RecordPoller id is " << poller_id_
            << ", target kafka pid is " << kafka_partition_id_
            << ", target kafka topic is " << opts.kafka_topic
            << ", kafka begin offset is " << ingested_offset_ + 1
            << ", init concurrency is " << concurrency_
            << ", max concurrency is " << max_concurrency_;
}

RecordPoller::~RecordPoller() {
  consumer_->unassign();
  CheckPendingFutures();
  assert(pending_futures_.empty());
}

bool RecordPoller::PollAndIngest() {
  // Check status of ingesting futures before polling.
  CheckPendingFutures();

  if (__builtin_expect(concur_mutated_.load(), false)) {
    concurrency_ = max_concurrency_;
    concur_mutated_.store(false);
  }

  // "concurrency_ == 0" means that there is no remaining concurrency
  // left for new ingesting tasks.
  if (concurrency_ == 0) {
    return false;
  }

  auto msg = consumer_->poll();
  if (!msg) {
    return false;
  }

  if (msg.get_error()) {
    LOG(WARNING) << "Error msg polled in kafka partition "
                 << kafka_partition_id_
                 << ": " << msg.get_error().to_string() << ", msg is ignore!";
    return !msg.is_eof();
  }

  IngestBatch(std::move(msg));
  return true;
}

void RecordPoller::CheckPendingFutures() {
  // Find the min ingested offset.
  while (!pending_futures_.empty()) {
    auto& front = pending_futures_.front();
    if (IsReady(front.fut)) {
    #ifdef DGS_BENCHMARK
      num_processed_records_ += front.fut.get();
    #else
      front.fut.get();
    #endif
      ingested_offset_.store(front.offset);
      if (!front.concurrency_released) {
        concurrency_ = std::min(concurrency_ + 1, max_concurrency_.load());
      }
      pending_futures_.pop_front();
    } else {
      break;
    }
  }
  // Check other pending futures to release concurrency.
  for (auto& info : pending_futures_) {
    if (!info.concurrency_released && IsReady(info.fut)) {
      concurrency_ = std::min(concurrency_ + 1, max_concurrency_.load());
      info.concurrency_released = true;
    }
  }
}

void RecordPoller::IngestBatch(cppkafka::Message&& msg) {
  if (!msg.get_payload().get_data()) {
    return;
  }
  auto offset = msg.get_offset();
  auto *data = const_cast<char*>(
      reinterpret_cast<const char*>(msg.get_payload().get_data()));
  auto size = msg.get_payload().get_size();
#ifdef DGS_BENCHMARK
  num_processed_bytes_ += size;
#endif
  auto buf = actor::BytesBuffer(
      data, size, seastar::make_object_deleter(std::move(msg)));
  auto fut = ingestor_(std::move(buf));
  pending_futures_.emplace_back(offset, std::move(fut));
  assert(concurrency_ > 0);
  concurrency_--;
}

bool RecordPoller::IsReady(const std::future<size_t>& fut) {
  return fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

RecordPollingWorker::RecordPollingWorker(const std::string& worker_name,
                                         uint32_t retry_interval_ms,
                                         std::vector<RecordPoller*>&& pollers)
  : worker_name_(worker_name),
    record_pollers_(std::move(pollers)),
    retry_interval_(retry_interval_ms),
    worker_thread_() {
}

RecordPollingWorker::~RecordPollingWorker() {
  if (!stopped_.load(std::memory_order_relaxed)) {
    Stop();
  }
}

void RecordPollingWorker::Start() {
  LOG(INFO) << "RecordPollingWorker is starting. "
            << "worker thread name is " << worker_name_;
  stopped_.store(false, std::memory_order_relaxed);
  worker_thread_ = std::make_unique<std::thread>(
    &RecordPollingWorker::PollForEach, this);
}

void RecordPollingWorker::Stop() {
  stopped_.store(true, std::memory_order_relaxed);
  if (worker_thread_) {
    worker_thread_->join();
  }
  LOG(INFO) << "RecordPollingWorker is stopped. "
            << "worker thread name is " << worker_name_;
}

void RecordPollingWorker::PollForEach() {
  LOG(INFO) << "worker thread " << worker_name_ << " is running";
  bool polled;
  while (true) {
    if (stopped_.load(std::memory_order_relaxed)) {
      return;
    }
    polled = false;
    for (auto* p : record_pollers_) {
      polled |= p->PollAndIngest();
    }
    if (!polled) {
      std::this_thread::sleep_for(retry_interval_);
    }
  }
  LOG(INFO) << "worker thread " << worker_name_ << " is stopped";
}

void
RecordPollingManager::Init(
    Partitioner&& partitioner,
    const std::vector<ShardId>& store_partition_routing_info,
    const std::vector<uint32_t>& kafka_partitions,
    const std::unordered_map<uint32_t, int64_t>& ready_offsets) {
  auto& opts = Options::GetInstance().GetRecordPollingOptions();
  partitioner_ = std::move(partitioner);
  partition_router_ = PartitionRouter(store_partition_routing_info);
  auto worker_type = Options::GetInstance().GetWorkerType();

  auto& kafka_topic = opts.kafka_topic;
  for (size_t i = 0; i < kafka_partitions.size(); ++i) {
    uint32_t kafka_pid = kafka_partitions[i];
    int64_t ingested_offset = (ready_offsets.count(kafka_pid) > 0)
        ? ready_offsets.at(kafka_pid) : -1;
    kafka_pid_to_pollers_.emplace_back(kafka_pid,
        std::make_unique<RecordPoller>(
            i, worker_type, opts, kafka_pid, ingested_offset,
            &partition_router_, &partitioner_));
  }
}

void RecordPollingManager::Start() {
  auto& opts = Options::GetInstance().GetRecordPollingOptions();
  auto worker_num = std::min(opts.thread_num,
      static_cast<uint32_t>(kafka_pid_to_pollers_.size()));

  // Assign pollers to each RecordPollingWorkers in a round-robin manner.
  std::vector<std::vector<RecordPoller*>> pollers_vec(worker_num);
  for (auto& iter : kafka_pid_to_pollers_) {
    pollers_vec[iter.first % worker_num].push_back(iter.second.get());
  }

  workers_.clear();
  workers_.reserve(worker_num);
  for (int i = 0; i < worker_num; ++i) {
    workers_.push_back(std::make_unique<RecordPollingWorker>(
      "record-polling-worker-" + std::to_string(i),
      opts.retry_interval_in_ms, std::move(pollers_vec[i])));
  }

  for (auto& worker : workers_) {
    worker->Start();
  }

#ifdef DGS_BENCHMARK
  bulkload_start_time_ += CurrentTimeInMs();
#endif

  LOG(INFO) << "RecordPollingManager is started.";
}

void RecordPollingManager::Stop() {
  LOG(INFO) << "RecordPollingManager is stopping.";
  for (auto& worker : workers_) {
    worker->Stop();
  }
  LOG(INFO) << "RecordPollingManager is stopped.";
}

void RecordPollingManager::BlockUntilReady() {
  auto& opts = Options::GetInstance().GetRecordPollingOptions();
  auto consumer = std::make_unique<cppkafka::Consumer>(cppkafka::Configuration{
    {"metadata.broker.list", opts.FormatKafkaServers()},
    {"broker.address.family", "v4"},
    {"group.id", "bulkload-complete-checker"},
    {"enable.auto.commit", false}});

  const size_t num_kafka_parts = kafka_pid_to_pollers_.size();

  std::vector<bool> kafka_part_is_ready(num_kafka_parts, false);
  std::vector<int64_t> ready_offsets(num_kafka_parts, 0);

  for (int i = 0; i < num_kafka_parts; ++i) {
    auto kafka_pid = kafka_pid_to_pollers_[i].first;
    bool offset_retrieved = false;
    int remaing_retry_times = 10;
    while (!offset_retrieved && remaing_retry_times-- > 0) {
      try {
        auto offsets = consumer->query_offsets(cppkafka::TopicPartition{
          opts.kafka_topic, static_cast<int>(kafka_pid)},
          std::chrono::milliseconds(30000));
        ready_offsets[i] = std::get<1>(offsets) - 1;
        offset_retrieved = true;

        LOG(INFO) << "Ready offset for kafka partition "
                << kafka_pid << " is " << ready_offsets[i];
      } catch (cppkafka::HandleException& ex) {
        LOG(WARNING) << "cppkafka::HandleException: "
                     << ex.get_error().to_string();
      }
    }

    if (!offset_retrieved) {
      LOG(FATAL) << "Ready offset for kafka partition "
                 << kafka_pid << " can't be queried";
    }
  }

  auto not_ready_count = num_kafka_parts;
  size_t not_ready_sleep_times = 0;
  while (not_ready_count > 0) {
    for (int i = 0; i < num_kafka_parts; ++i) {
      if (!kafka_part_is_ready[i]) {
        auto& poller = kafka_pid_to_pollers_[i].second;
        auto cur_offset = poller->GetIngestedOffset();
        if (cur_offset >= ready_offsets[i]) {
          kafka_part_is_ready[i] = true;
          --not_ready_count;
        }
      }
    }

    if (++not_ready_sleep_times % 500 == 0) {
      for (int i = 0; i < num_kafka_parts; ++i) {
        auto &poller_pair = kafka_pid_to_pollers_[i];
        auto& poller = poller_pair.second;
        auto cur_offset = poller->GetIngestedOffset();
        LOG(INFO) << "record poller " << i << " is not ready"
                  << ", target kafka pid is " << poller_pair.first
                  << ", current ingested offset is " << cur_offset;
      }
    }

    // sleep every 200 milliseconds.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

#ifdef DGS_BENCHMARK
  bulkload_end_time_ = CurrentTimeInMs();
  uint64_t total_records = 0;
  uint64_t total_bytes = 0;
  for (auto& pp : kafka_pid_to_pollers_) {
    std::cout << "#processed records for kafka partition " << pp.first << ": "
              << pp.second->num_processed_records() << std::endl;
    total_records += pp.second->num_processed_records();
    total_bytes += pp.second->num_processed_bytes();
  }

  // milliseconds
  auto elapsed_time = bulkload_end_time_ - bulkload_start_time_;

  LOG(INFO) << "Bulk loading elapsed time is " << elapsed_time
            << " millseconds";
  LOG(INFO) << "Bulk loading throughput is " << std::fixed
            << std::setprecision(0)
            << float(total_records) / float(elapsed_time) * 1000
            << " records per second.";
  LOG(INFO) << "Bulk loading throughput is " << std::fixed
            << std::setprecision(0)
            << float(total_bytes) / float(1024) / float(elapsed_time) * 1000
            << " KB per second" << std::endl;
#endif
}

void RecordPollingManager::SetConcurrency(uint32_t value) {
  for (auto &poller_pair : kafka_pid_to_pollers_) {
    auto& poller = poller_pair.second;
    poller->SetMaxConcurrency(value);
  }
  LOG(INFO) << "Kafka polling concurrency is set to " << value;
}

void RecordPollingManager::GetConcurrency() {
  for (auto &poller_pair : kafka_pid_to_pollers_) {
    auto& poller = poller_pair.second;
    LOG(INFO) << "Kafka pid: " << poller_pair.first
              << ", current max concurrency: "
              << poller->GetMaxConcurrency();
  }
}

std::unordered_map<uint32_t, int64_t>
RecordPollingManager::GetIngestedOffsets() {
  std::unordered_map<uint32_t, int64_t> ingested_offsets;
  for (auto& poller_pair : kafka_pid_to_pollers_) {
    ingested_offsets[poller_pair.first] =
        poller_pair.second->GetIngestedOffset();
  }
  return ingested_offsets;
}

}  // namespace dgs
