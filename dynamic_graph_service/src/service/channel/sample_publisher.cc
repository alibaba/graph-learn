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

#include "service/channel/sample_publisher.h"

#include "seastar/core/alien.hh"
#include "seastar/core/when_all.hh"

#include "common/actor_wrapper.h"
#include "common/log.h"
#include "common/options.h"
#include "common/utils.h"
#include "core/io/sample_update_batch.h"

namespace dgs {

ProducingCallbackUnit::ProducingCallbackUnit(
    const std::shared_ptr<actor::VoidPromiseManager>& pr_manager,
    uint32_t pr_id, uint32_t lshard_id, uint32_t retry_times)
  : pr_manager(pr_manager), pr_id(pr_id),
    lshard_id(lshard_id), retry_times(retry_times) {
}

void KafkaProducerPool::Init() {
  auto& opts = Options::GetInstance().GetSamplePublishingOptions();
  producers_.reserve(opts.producer_pool_size);
  for (int i = 0; i < opts.producer_pool_size; i++) {
    cppkafka::Configuration conf{
      {"metadata.broker.list", opts.FormatKafkaServers()},
      {"broker.address.family", "v4"}};
    conf.set_delivery_report_callback(
        [] (cppkafka::Producer& producer, const cppkafka::Message& msg) {
      auto* cb_unit = static_cast<ProducingCallbackUnit*>(msg.get_user_data());
      if (__builtin_expect(msg.get_error() && true, false)) {
        if (cb_unit->retry_times > 0) {
          cb_unit->retry_times--;
          producer.produce(msg);
        } else {
          seastar::alien::run_on(cb_unit->lshard_id, [cb_unit] {
            cb_unit->pr_manager->set_exception(cb_unit->pr_id,
              std::make_exception_ptr(kafka_produce_exception()));
            delete cb_unit;
          });
        }
        return;
      }
      seastar::alien::run_on(cb_unit->lshard_id, [cb_unit] {
        cb_unit->pr_manager->set_value(cb_unit->pr_id);
        delete cb_unit;
      });
    });
    auto producer = std::make_shared<cppkafka::Producer>(std::move(conf));
    producers_.emplace_back(std::move(producer));
  }
  cb_poller_ = std::thread(&KafkaProducerPool::PollCallback, this,
                           opts.cb_poll_interval_in_ms);
}

void KafkaProducerPool::Finalize() {
  stopped_.store(true, std::memory_order_relaxed);
  if (cb_poller_.joinable()) {
    cb_poller_.join();
  }
  FlushAll();
}

void KafkaProducerPool::FlushAll() {
  for (auto& p : producers_) {
    p->flush();
  }
}

void KafkaProducerPool::PollCallback(uint32_t interval) {
  const std::chrono::milliseconds cb_polling_timeout(0);
  const std::chrono::milliseconds cb_polling_interval(interval);
  while (true) {
    if (stopped_.load(std::memory_order_relaxed)) {
      return;
    }
    for (auto& p : producers_) {
      p->poll(cb_polling_timeout);
    }
    std::this_thread::sleep_for(cb_polling_interval);
  }
}

SamplePublisher::SamplePublisher() {
  auto& opts = Options::GetInstance().GetSamplePublishingOptions();
  kafka_topic_ = opts.kafka_topic;
  kafka_partition_num_ = opts.kafka_partition_num;
  retry_times_ = opts.max_produce_retry_times;
  pr_manager_ = std::make_shared<actor::VoidPromiseManager>(1024);
  auto* producer_pool = KafkaProducerPool::GetInstance();
  auto producer_idx = actor::LocalShardId() % producer_pool->Size();
  kafka_producer_ = producer_pool->GetProducer(producer_idx);
}

void SamplePublisher::UpdateDSPublishInfo(
    uint32_t serving_worker_num,
    const std::vector<uint32_t>& kafka_to_serving_worker_vec) {
  assert(kafka_to_serving_worker_vec.size() == kafka_partition_num_);
  worker_records_.resize(serving_worker_num);
  worker_kafka_routers_.resize(serving_worker_num);
  for (uint32_t i = 0; i < kafka_to_serving_worker_vec.size(); i++) {
    auto wid = kafka_to_serving_worker_vec[i];
    worker_kafka_routers_[wid].kafka_pids.push_back(i);
  }
}

seastar::future<> SamplePublisher::Publish(
    const std::vector<storage::KVPair>& batch,
    const std::vector<storage::SubsInfo>& infos) {
  if (batch.empty() || infos.empty()) {
    return seastar::make_ready_future<>();
  }

  std::vector<uint32_t> pr_ids;
  std::vector<seastar::future<>> futures;
  cppkafka::MessageBuilder builder(kafka_topic_);

  std::vector<size_t> worker_batch_size(worker_records_.size(), 0);
  for (auto& info : infos) {
    auto wid = info.worker_id;
    auto rid = info.record_id;
    assert(wid < worker_records_.size());
    worker_records_[wid].emplace_back(&batch[rid]);
    worker_batch_size[wid] += batch[rid].Size();
    if (worker_batch_size[wid] > batch_size_) {
      auto pr_id = ProduceWorkerUpdates(wid, worker_records_[wid], builder);
      pr_ids.push_back(pr_id);
      futures.emplace_back(pr_manager_->get_future(pr_id));
      worker_records_[wid].clear();
      worker_batch_size[wid] = 0;
    }
  }
  for (WorkerId wid = 0; wid < worker_records_.size(); ++wid) {
    if (!worker_records_[wid].empty()) {
      auto pr_id = ProduceWorkerUpdates(wid, worker_records_[wid], builder);
      pr_ids.push_back(pr_id);
      futures.emplace_back(pr_manager_->get_future(pr_id));
      worker_records_[wid].clear();
    }
  }

  // Return when all the producing tasks are completed
  return seastar::when_all(futures.begin(), futures.end()).then(
      [pr_ids = std::move(pr_ids), this] (const std::vector<seastar::future<>>& ret) {
    for (auto pr_id : pr_ids) {
      pr_manager_->remove_pr(pr_id);
    }
    for (auto& r : ret) {
      if (r.failed()) {
        // TODO(@goldenleaves): Return detailed failing results
        return seastar::make_exception_future<>(
            kafka_produce_exception());
      }
    }
    return seastar::make_ready_future<>();
  });
}

uint32_t SamplePublisher::ProduceWorkerUpdates(
    WorkerId wid,
    const WorkerSampleUpdates& updates,
    cppkafka::MessageBuilder& builder) {
  auto dst_kafka_pid = worker_kafka_routers_[wid].GetKafkaPid();
  builder.partition(static_cast<int>(dst_kafka_pid));
  auto buf = io::SampleUpdateBatch::Serialize(updates.data(), updates.size());
  builder.payload({buf.get(), buf.size()});
  // Get promise and set callback unit
  auto pr_id = pr_manager_->acquire_pr();
  builder.user_data(new ProducingCallbackUnit(
      pr_manager_, pr_id, actor::LocalShardId(), retry_times_));
  // Produce
  kafka_producer_->produce(builder);
  return pr_id;
}

}  // namespace dgs
