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

#ifndef DGS_SERVICE_CHANNEL_SAMPLE_PUBLISHER_H_
#define DGS_SERVICE_CHANNEL_SAMPLE_PUBLISHER_H_

#include <exception>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cppkafka/producer.h"

#include "common/actor_wrapper.h"
#include "core/storage/sample_store.h"
#include "core/storage/subscription_table.h"

namespace dgs {

class kafka_produce_exception : public std::exception {
  const char* what() const noexcept override {
    return "kafka produce fails!";
  }
};

struct ProducingCallbackUnit {
  std::shared_ptr<actor::VoidPromiseManager> pr_manager;
  uint32_t pr_id;
  uint32_t lshard_id;
  uint32_t retry_times;

  ProducingCallbackUnit(
    const std::shared_ptr<actor::VoidPromiseManager>& pr_manager,
    uint32_t pr_id, uint32_t lshard_id, uint32_t retry_times);
  ~ProducingCallbackUnit() = default;
};

/// A \KafkaProducerPool corresponds holds kafka producers which can be
/// used to push data to the designated kafka servers
///
/// Note that, kafka producers are thread-safe.
class KafkaProducerPool {
  using ProducerPtr = std::shared_ptr<cppkafka::Producer>;

public:
  static KafkaProducerPool* GetInstance();

  void Init();
  void Finalize();

  ProducerPtr GetProducer(uint32_t index) {
    return producers_[index];
  }

  void FlushAll();

  uint32_t Size() {
    return producers_.size();
  }

private:
  KafkaProducerPool() = default;
  void PollCallback(uint32_t interval);

private:
  std::thread cb_poller_;
  std::atomic<bool> stopped_ = { false };
  std::vector<ProducerPtr> producers_;
};

inline
KafkaProducerPool* KafkaProducerPool::GetInstance() {
  static KafkaProducerPool instance;
  return &instance;
}

/// SamplePublisher can be used to publish sampled data to the designated
/// kafka topic using the producer in the \KafkaProducerPool
class SamplePublisher {
public:
  SamplePublisher() = default;
  SamplePublisher(const std::string& kafka_topic,
                  uint32_t kafka_partition_num);

  ~SamplePublisher() = default;

  SamplePublisher(SamplePublisher&& other) noexcept;
  SamplePublisher& operator=(SamplePublisher&& other) noexcept;

  seastar::future<> Publish(const std::vector<storage::KVPair>& batch,
                            const std::vector<storage::SubsInfo>& infos);

  void UpdateSinkKafkaPartitions(const std::vector<uint32_t>& updates,
                                 const std::string& serving_store_part_strategy,
                                 uint32_t serving_store_part_num,
                                 uint32_t serving_worker_num);

private:
  // FIXME(@xmqin): use flat vector instead.
  using WorkerWisePartitions = std::vector<std::vector<const storage::KVPair*>>;

private:
  std::string            kafka_topic_;
  uint32_t               kafka_partition_num_ = 0;
  uint32_t               retry_times_ = 0;
  Partitioner            partitioner_;
  std::vector<std::vector<uint32_t>>         worker_sink_kafka_partitions_;
  std::vector<WorkerWisePartitions>          worker_records_;
  std::vector<std::unordered_set<uint32_t>>  worker_record_id_set_;
  std::shared_ptr<actor::VoidPromiseManager> pr_manager_;
  std::shared_ptr<cppkafka::Producer>        kafka_producer_;
};

}  // namespace dgs

#endif  // DGS_SERVICE_CHANNEL_SAMPLE_PUBLISHER_H_
