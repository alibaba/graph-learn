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

#include "log_poller.h"

#include <chrono>
#include <memory>

#include "gs_option.h"

namespace dgs {
namespace dataloader {
namespace gs {

LogPoller::LogPoller(int32_t kafka_partition, DataStreamOffset start_offset, const std::string& meta_file)
  : consumer_(cppkafka::Configuration{
      {"metadata.broker.list", GSOptions::GetInstance().FormatSourceKafkaBrokers()},
      {"broker.address.family", "v4"},
      {"group.id", "graph-store-log-poller"},
      {"enable.auto.commit", false}}),
    retry_interval_(GSOptions::GetInstance().polling_retry_ms),
    persist_interval_ms_(GSOptions::GetInstance().polling_offset_persist_ms),
    batch_size_(Options::GetInstance().output_batch_size),
    data_partition_num_(Options::GetInstance().data_partitions),
    flush_interval_ms_(GSOptions::GetInstance().polling_flush_ms) {
  // set consumer
  consumer_.assign({cppkafka::TopicPartition{
      GSOptions::GetInstance().source_kafka_topic, kafka_partition, start_offset}});
  consumer_.set_timeout(std::chrono::milliseconds(0));
  // init producing builders
  builders_.reserve(data_partition_num_);
  for (uint32_t i = 0; i < data_partition_num_; i++) {
    builders_.emplace_back(i);
  }
  // init offsets
  persisted_offset_ = start_offset - 1;
  current_offset_ = persisted_offset_;
  last_persist_time_ = CurrentTimeInMs();
  offset_writer_.open(meta_file, std::ios::trunc);
  if (!offset_writer_.good()) {
    throw std::runtime_error("Failed to open polling offset writer with path: " + meta_file);
  }
  // init flushing info list
  flush_infos_.Init(data_partition_num_, persisted_offset_);
}

LogPoller::~LogPoller() {
  StopPolling();
  offset_writer_.close();
}

void LogPoller::StartPolling() {
  stopped_.store(false);
  worker_ = std::thread(&LogPoller::Poll, this);
}

void LogPoller::StopPolling() {
  stopped_.store(true);
  if (worker_.joinable()) {
    worker_.join();
  }
  // Flush the remaining data
  FlushAllPending();
  // Do offset persisting
  PersistConsumingProgress();
}

void LogPoller::Poll() {
  while (true) {
    if (stopped_.load(std::memory_order_relaxed)) {
      return;
    }
    auto msg = consumer_.poll();
    if (msg) {
      ProcessLogMessage(msg);
    } else {
      std::this_thread::sleep_for(retry_interval_);
    }
    // Doing checks after each processing of polled message.
    CheckAndFlush();
    CheckAndPersist();
  }
}

int64_t LogPoller::CurrentTimeInMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

void LogPoller::ProcessLogMessage(const cppkafka::Message& msg) {
  if (msg.get_error()) {
    return;
  }
  polled_num_++;
  current_offset_ = msg.get_offset();
  lgraph::log_subscription::MessageParser parser(msg.get_payload().get_data(), msg.get_payload().get_size());
  auto operations = parser.GetOperations();
  for (auto& op : operations) {
    auto op_type = op.GetOpType();
    // Only process log messages related to vertex/edge updates
    if (op_type == lgraph::OVERWRITE_VERTEX || op_type == lgraph::UPDATE_VERTEX) {
      ProcessVertexUpdate(op.GetInfoAsVertexInsertOp());
    } else if (op_type == lgraph::OVERWRITE_EDGE || op_type == lgraph::UPDATE_EDGE) {
      ProcessEdgeUpdate(op.GetInfoAsEdgeInsertOp());
    }
  }
}

void LogPoller::ProcessVertexUpdate(const lgraph::log_subscription::VertexInsertInfo& info) {
  auto data_pid = Partitioner::GetInstance().GetDataPartitionId(info.GetVertexId());
  auto& bb = builders_.at(data_pid).bb;
  bb.AddVertexUpdate(info);
  if (bb.RecordNum() >= batch_size_) {
    FlushPending(data_pid);
  }
}

void LogPoller::ProcessEdgeUpdate(const lgraph::log_subscription::EdgeInsertInfo& info) {
  if (!info.IsForward()) {
    return;
  }
  auto edge_id = info.GetEdgeId();
  auto data_pid = Partitioner::GetInstance().GetDataPartitionId(edge_id.src_vertex_id);
  auto& bb = builders_.at(data_pid).bb;
  bb.AddEdgeUpdate(info);
  if (bb.RecordNum() >= batch_size_) {
    FlushPending(data_pid);
  }
}

void LogPoller::CheckAndFlush() {
  while (CurrentTimeInMs() - flush_infos_.Head().time >= flush_interval_ms_) {
    FlushPending(flush_infos_.Head().idx);
  }
}

void LogPoller::FlushPending(uint32_t idx) {
  auto& builder = builders_.at(idx);
  if (builder.bb.RecordNum() > 0) {
    builder.bb.Finish();
    builder.SetPayload();
    producer_.SyncProduce(builder.mb);
    builder.bb.Clear();
  }
  // Update the time and consuming offsets after each flush
  flush_infos_.Update(idx, current_offset_);
}

void LogPoller::FlushAllPending() {
  for (uint32_t i = 0; i < data_partition_num_; i++) {
    FlushPending(i);
  }
}

void LogPoller::CheckAndPersist() {
  if ((CurrentTimeInMs() - last_persist_time_) >= persist_interval_ms_) {
    PersistConsumingProgress();
  }
}

void LogPoller::PersistConsumingProgress() {
  // The log polling progress offset to persist should be the minimal val of
  // all output queues' recorded consuming offsets.
  auto offset = flush_infos_.Head().offset;
  if (offset > persisted_offset_) {
    offset_writer_.seekp(0, offset_writer_.beg);
    offset_writer_ << offset;
    offset_writer_.flush();
    persisted_offset_ = offset;
  }
  last_persist_time_ = CurrentTimeInMs();
}

LogPoller::Builder::Builder(uint32_t data_pid)
  : bb(data_pid), mb(Options::GetInstance().output_kafka_topic) {
  mb.partition(static_cast<int32_t>(Partitioner::GetInstance().GetKafkaPartitionId(data_pid)));
}

void LogPoller::OrderedFlushInfoList::Init(uint32_t num, DataStreamOffset init_offset) {
  infos_.clear();
  infos_.reserve(num);
  head_.next = &tail_;
  tail_.prev = &head_;
  auto cur_time = CurrentTimeInMs();
  for (uint32_t i = 0; i < num; i++) {
    infos_.emplace_back(i, init_offset, cur_time);
    auto* info = &infos_.at(i);
    InsertToTail(info);
  }
}

void LogPoller::OrderedFlushInfoList::Update(uint32_t idx, DataStreamOffset offset) {
  auto* info = &infos_.at(idx);
  info->offset = offset;
  info->time = CurrentTimeInMs();
  // Adjust position
  auto* prev = info->prev;
  auto* next = info->next;
  prev->next = next;
  next->prev = prev;
  InsertToTail(info);
}

void LogPoller::OrderedFlushInfoList::InsertToTail(FlushInfo* info) {
  auto* tail = tail_.prev;
  tail->next = info;
  info->prev = tail;
  tail_.prev = info;
  info->next = &tail_;
}


void LogPollingManager::Init(const PartitionOffsetMap& partition_to_offset) {
  auto& dl_opts = Options::GetInstance();
  auto& gs_opts = GSOptions::GetInstance();
  for (int32_t i = 0; i < gs_opts.source_kafka_partitions; i++) {
    // calculate the data source partitions to poll
    if ((i % dl_opts.loader_num) == dl_opts.loader_id) {
      std::string meta_file = gs_opts.polling_meta_dir + "/" + std::to_string(i);
      auto offset = std::max(RecoverOffset(meta_file), (partition_to_offset.count(i) > 0) ? partition_to_offset.at(i) : 0);
      pollers_.emplace(i, new LogPoller(i, offset, meta_file));
    }
  }
  LOG(INFO) << "Log pollers are inited!";
}

void LogPollingManager::Start() {
  for (auto& entry : pollers_) {
    entry.second->StartPolling();
  }
  LOG(INFO) << "Log polling is started!";
}

void LogPollingManager::Stop() {
  for (auto& entry : pollers_) {
    entry.second->StopPolling();
  }
  LOG(INFO) << "Log polling is stopped!";
}

void LogPollingManager::Finalize() {
  pollers_.clear();
  LOG(INFO) << "Log pollers are finalized!";
}

size_t LogPollingManager::PolledNum() const {
  size_t num = 0;
  for (auto& entry : pollers_) {
    num += entry.second->PolledNum();
  }
  return num;
}

DataStreamOffset LogPollingManager::RecoverOffset(const std::string& offset_file) {
  DataStreamOffset offset = 0;
  std::ifstream infile(offset_file);
  if (infile.good()) {
    infile >> offset;
  }
  infile.close();
  return offset;
}

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs
