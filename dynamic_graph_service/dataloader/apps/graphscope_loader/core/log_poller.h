/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHSCOPE_LOADER_CORE_LOG_POLLER_H_
#define GRAPHSCOPE_LOADER_CORE_LOG_POLLER_H_

#include <thread>

#include "dataloader/batch_producer.h"
#include "dataloader/partitioner.h"

#include "boost/lockfree/queue.hpp"
#include "lgraph/log_subscription/subscriber.h"

#include "gs_batch_builder.h"

namespace dgs {
namespace dataloader {
namespace gs {

/// A log poller thread corresponding to a single data source partition.
class LogPoller {
public:
  LogPoller(int32_t kafka_partition, DataStreamOffset start_offset, const std::string& meta_file);
  ~LogPoller();

  /// Start polling, this will create a new thread to run the polling while loop.
  void StartPolling();

  /// Stop polling and join the worker thread.
  void StopPolling();

  /// Get the polled message number of this poller.
  size_t PolledNum() const { return polled_num_; }

private:
  void Poll();
  static int64_t CurrentTimeInMs();
  // Process polled message
  void ProcessLogMessage(const cppkafka::Message& msg);
  void ProcessVertexUpdate(const lgraph::log_subscription::VertexInsertInfo& info);
  void ProcessEdgeUpdate(const lgraph::log_subscription::EdgeInsertInfo& info);
  // Check whether the interval from the last flush exceeds `flush_interval_ms_`
  // and decide from this to execute a flush immediately.
  void CheckAndFlush();
  void FlushPending(uint32_t idx);
  void FlushAllPending();
  // Check whether the interval from the last polling progress persisting exceeds `persist_interval_ms_`
  // and decide from this to execute a meta persist immediately.
  void CheckAndPersist();
  void PersistConsumingProgress();

private:
  std::thread worker_;
  std::atomic<bool> stopped_ = { false };

  // log polling
  cppkafka::Consumer consumer_;
  const std::chrono::milliseconds retry_interval_;
  size_t polled_num_ = 0;

  // batching && producing
  BatchProducer producer_;
  const uint32_t batch_size_;
  const uint32_t data_partition_num_;
  struct Builder {
    GSBatchBuilder bb;
    cppkafka::MessageBuilder mb;
    explicit Builder(uint32_t data_pid);
    void SetPayload() { mb.payload({bb.GetBufferPointer(), bb.GetBufferSize()}); }
  };
  std::vector<Builder> builders_;

  // polling offset persisting
  const uint32_t persist_interval_ms_;
  DataStreamOffset persisted_offset_;
  DataStreamOffset current_offset_;
  int64_t last_persist_time_;
  std::ofstream offset_writer_;

  // flushing
  const uint32_t flush_interval_ms_;
  struct FlushInfo {
    uint32_t idx = 0;
    DataStreamOffset offset = 0;
    int64_t time = 0;
    FlushInfo* prev = nullptr;
    FlushInfo* next = nullptr;
    FlushInfo() = default;
    FlushInfo(uint32_t idx, DataStreamOffset offset, int64_t time) : idx(idx), offset(offset), time(time) {}
  };
  class OrderedFlushInfoList {
  public:
    OrderedFlushInfoList() = default;
    void Init(uint32_t num, DataStreamOffset init_offset);
    void Update(uint32_t idx, DataStreamOffset offset);
    FlushInfo& Head() { return *(head_.next); }
  private:
    std::vector<FlushInfo> infos_;
    FlushInfo head_;
    FlushInfo tail_;
    void InsertToTail(FlushInfo* info);
  };
  OrderedFlushInfoList flush_infos_{};
};

class LogPollingManager {
public:
  using PartitionOffsetMap = std::unordered_map<int32_t, DataStreamOffset>;

  static LogPollingManager& GetInstance();

  /// Init the log pollers.
  ///
  /// If the start offset of a source kafka partition is not specified, this
  /// manager will try to recover it from persisted meta file.
  void Init(const PartitionOffsetMap& partition_to_offset = {});

  /// StartPolling the log polling.
  void Start();

  /// Stop the log polling.
  void Stop();

  /// Delete all the pollers.
  void Finalize();

  /// Get the total polled message number of all pollers.
  size_t PolledNum() const;

private:
  static DataStreamOffset RecoverOffset(const std::string& offset_file);

private:
  using PollerPtr = std::unique_ptr<LogPoller>;
  std::unordered_map<int32_t, PollerPtr> pollers_;
};

inline
LogPollingManager& LogPollingManager::GetInstance() {
  static LogPollingManager instance;
  return instance;
}

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs

#endif // GRAPHSCOPE_LOADER_CORE_LOG_POLLER_H_
