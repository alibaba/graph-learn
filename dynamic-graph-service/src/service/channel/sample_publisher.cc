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
    if (i == 0) {
      auto config_map = producer->get_configuration().get_all();
      for (auto iter : config_map) {
        LOG(INFO) << "[kafka produer configuration] "
                  << iter.first << ", " << iter.second;
      }
    }
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

SamplePublisher::SamplePublisher(const std::string& kafka_topic,
                                 uint32_t kafka_partition_num)
  : kafka_topic_(kafka_topic),
    kafka_partition_num_(kafka_partition_num) {
  auto& opts = Options::GetInstance().GetSamplePublishingOptions();
  retry_times_ = opts.max_produce_retry_times;
  pr_manager_ = std::make_shared<actor::VoidPromiseManager>(1024);
  auto* producer_pool = KafkaProducerPool::GetInstance();
  auto producer_idx = actor::LocalShardId() % producer_pool->Size();
  kafka_producer_ = producer_pool->GetProducer(producer_idx);
}

SamplePublisher::SamplePublisher(SamplePublisher&& other) noexcept
  : kafka_topic_(std::move(other.kafka_topic_)),
    kafka_partition_num_(other.kafka_partition_num_),
    worker_sink_kafka_partitions_(other.worker_sink_kafka_partitions_),
    retry_times_(other.retry_times_),
    worker_records_(std::move(other.worker_records_)),
    pr_manager_(std::move(other.pr_manager_)),
    kafka_producer_(std::move(other.kafka_producer_)) {
  other.kafka_topic_ = "";
  other.kafka_partition_num_ = 0;
  other.retry_times_ = 0;
}

SamplePublisher& SamplePublisher::operator=(SamplePublisher&& other) noexcept {
  if (this != &other) {
    kafka_topic_ = std::move(other.kafka_topic_);
    kafka_partition_num_ = other.kafka_partition_num_;
    worker_sink_kafka_partitions_ = other.worker_sink_kafka_partitions_;
    retry_times_ = other.retry_times_;
    worker_records_ = std::move(other.worker_records_);
    pr_manager_ = std::move(other.pr_manager_);
    kafka_producer_ = std::move(other.kafka_producer_);
    other.kafka_topic_ = "";
    other.kafka_partition_num_ = 0;
    other.retry_times_ = 0;
  }
  return *this;
}

void SamplePublisher::UpdateSinkKafkaPartitions(
    const std::vector<uint32_t>& updates,
    const std::string& serving_store_part_strategy,
    uint32_t serving_store_part_num,
    uint32_t serving_worker_num) {
  try {
    auto partitioner = PartitionerFactory::Create(
        serving_store_part_strategy, serving_store_part_num);
    partitioner_ = std::move(partitioner);
    LOG(INFO) << "Update serving store partition info with strategy: "
               << serving_store_part_strategy << ", partition num: "
               << serving_store_part_num;
  } catch (std::exception& ex) {
    LOG(ERROR) << "Update serving store partition info failed: " << ex.what();
  }

  const auto downstream_store_pids_num = updates.size();

  worker_records_.reserve(serving_worker_num);
  worker_sink_kafka_partitions_.reserve(serving_worker_num);
  for (int i = 0; i < serving_worker_num; ++i) {
    // FIXME(@xmqin): using per worker specification.
    worker_sink_kafka_partitions_.push_back(updates);
    worker_record_id_set_.emplace_back(std::unordered_set<uint32_t>());

    // index: downstream storage partition id.
    // value: target kafka partition id.
    WorkerWisePartitions part_records;
    for (size_t j = 0; j < downstream_store_pids_num; j++) {
      part_records.emplace_back(std::vector<const storage::KVPair*>{});
    }
    worker_records_.emplace_back(std::move(part_records));
  }
}

seastar::future<> SamplePublisher::Publish(
    const std::vector<storage::KVPair>& batch,
    const std::vector<storage::SubsInfo>& infos) {
  if (batch.empty() || infos.empty()) {
    return seastar::make_ready_future<>();
  }

  for (auto &info : infos) {
    assert(info.worker_id < worker_records_.size());
    if (!worker_record_id_set_[info.worker_id].count(info.record_id)) {
      auto dst_pid = partitioner_.GetPartitionId(
        batch[info.record_id].key.pkey.vid);
      worker_records_[info.worker_id][dst_pid].push_back(
        &batch[info.record_id]);
      worker_record_id_set_[info.worker_id].insert(info.record_id);
    }
  }

  std::vector<uint32_t> pr_ids;
  std::vector<seastar::future<>> futures;
  cppkafka::MessageBuilder builder(kafka_topic_);

  for (WorkerId wid = 0; wid < worker_records_.size(); ++wid) {
    auto& part_records = worker_records_[wid];
    auto& sink_kafka_partitions = worker_sink_kafka_partitions_[wid];
    for (PartitionId pid = 0; pid < part_records.size(); ++pid) {
      if (part_records[pid].empty()) {
        continue;
      }
      // Get the sink kafka partition id
      // TODO(@xmqin): double-check
      auto dst_kafka_pid = sink_kafka_partitions[pid];
      builder.partition(static_cast<int>(dst_kafka_pid));
      // Set payload with SampleUpdateBatch storing KVPairs.
      io::SampleUpdateBatch update_batch(pid, part_records[pid]);
      // TODO(@xmqin): reserve capacity.
      part_records[pid].clear();
      // FIXME(@xmqin): try to reduce memcpy
      builder.payload({update_batch.Data(), update_batch.Size()});
      if (update_batch.Size() > 1024 * 1024) {
        LOG(WARNING) << "data batch size is " << batch.size()
                     << ", subs info size is " << infos.size()
                     << ", kafka message size: "
                     << static_cast<float>(update_batch.Size()) / 1000.0
                     << " KB, " << "dst kafka pid: " << pid;
      }
      // Get promise and set callback unit
      auto pr_id = pr_manager_->acquire_pr();
      pr_ids.push_back(pr_id);
      futures.emplace_back(pr_manager_->get_future(pr_id));
      builder.user_data(new ProducingCallbackUnit(
      pr_manager_, pr_id, actor::LocalShardId(), retry_times_));
      // Produce
      kafka_producer_->produce(builder);
    }
    worker_record_id_set_[wid].clear();
  }

  // Return once all the sending tasks are completed
  return seastar::when_all(futures.begin(), futures.end()).then(
      [pr_ids = std::move(pr_ids), this] (const std::vector<seastar::future<>>& ret) {
    for (auto pr_id : pr_ids) {
      pr_manager_->remove_pr(pr_id);
    }
    for (auto& r : ret) {
      if (r.failed()) {
        // TODO(xiaoming): Return detailed failing results
        return seastar::make_exception_future<>(
            kafka_produce_exception());
      }
    }
    return seastar::make_ready_future<>();
  });
}

}  // namespace dgs
