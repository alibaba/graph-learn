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

#include "bulk_loader.h"

#include "lgraph/db/readonly_db.h"
#include "dataloader/logging.h"

#include "gs_batch_builder.h"
#include "gs_option.h"

namespace dgs {
namespace dataloader {
namespace gs {

void BulkLoadingThreadPool::Init() {
  if (workers_ == nullptr) {
    workers_ = std::make_unique<boost::asio::thread_pool>(GSOptions::GetInstance().bulk_loading_threads);
  }
  LOG(INFO) << "Bulk loading thread pool is inited!";
}

void BulkLoadingThreadPool::Join() {
  if (workers_ != nullptr) {
    workers_->join();
  }
  LOG(INFO) << "Bulk loading threads are joined!";
}

void BulkLoadingThreadPool::Finalize() {
  workers_.reset();
  LOG(INFO) << "Bulk loading thread pool are finalized!";
}

void BulkLoadingThreadPool::Load(const BulkLoadingInfo& loading_info, Callback callback) {
  if (workers_ == nullptr) {
    LOG(ERROR) << "The bulk loading thread pool is not inited!";
    return;
  }
  boost::asio::post(*workers_, [info = loading_info, cb = std::move(callback)] {
    DoLoad(info, cb);
  });
}

void BulkLoadingThreadPool::DoLoad(const BulkLoadingInfo& info, Callback callback) {
  auto& partitioner = Partitioner::GetInstance();
  auto& opts = Options::GetInstance();
  auto batch_size = opts.output_batch_size;
  auto data_partition_num = opts.data_partitions;

  std::vector<GSBatchBuilder> batch_builders;
  batch_builders.reserve(data_partition_num);
  std::vector<cppkafka::MessageBuilder> msg_builders;
  msg_builders.reserve(data_partition_num);
  for (uint32_t i = 0; i < data_partition_num; i++) {
    batch_builders.emplace_back(i);
    msg_builders.emplace_back(opts.output_kafka_topic);
    msg_builders[i].partition(static_cast<int32_t>(partitioner.GetKafkaPartitionId(i)));
  }

  auto db = lgraph::db::ReadonlyDB::Open(info.db_name.c_str());
  auto snapshot = db.GetSnapshot(info.snapshot_id);
  auto schema = lgraph::Schema::FromProtoFile(info.schema_file.c_str());

  BatchProducer producer;

  // Load vertices
  auto v_iter = snapshot.ScanVertex().unwrap();
  assert(v_iter.Valid());
  while (true) {
    auto v = v_iter.Next().unwrap();
    if (!v.Valid()) {
      break;
    }
    auto data_pid = partitioner.GetDataPartitionId(v.GetVertexId());
    auto& bb = batch_builders.at(data_pid);
    bb.AddVertexUpdate(&v, schema);
    if (bb.RecordNum() >= batch_size) {
      bb.Finish();
      auto& mb = msg_builders.at(data_pid);
      mb.payload({bb.GetBufferPointer(), bb.GetBufferSize()});
      producer.SyncProduce(mb);
      bb.Clear();
    }
  }

  // Load edges
  auto e_iter = snapshot.ScanEdge().unwrap();
  assert(e_iter.Valid());
  while (true) {
    auto e = e_iter.Next().unwrap();
    if (!e.Valid()) {
      break;
    }
    auto edge_id = e.GetEdgeId();
    auto data_pid = partitioner.GetDataPartitionId(edge_id.src_vertex_id);
    auto& bb = batch_builders.at(data_pid);
    bb.AddEdgeUpdate(&e, schema);
    if (bb.RecordNum() >= batch_size) {
      bb.Finish();
      auto& mb = msg_builders.at(data_pid);
      mb.payload({bb.GetBufferPointer(), bb.GetBufferSize()});
      producer.SyncProduce(mb);
      bb.Clear();
    }
  }

  // Flush remaining
  for (uint32_t i = 0; i < data_partition_num; i++) {
    auto& bb = batch_builders.at(i);
    if (bb.RecordNum() > 0) {
      bb.Finish();
      auto& mb = msg_builders.at(i);
      mb.payload({bb.GetBufferPointer(), bb.GetBufferSize()});
      producer.SyncProduce(mb);
      bb.Clear();
    }
  }

  // Execute callback
  callback();
}

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs
