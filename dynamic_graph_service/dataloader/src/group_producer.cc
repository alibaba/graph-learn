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

#include "dataloader/group_producer.h"

#include "dataloader/partitioner.h"

namespace dgs {
namespace dataloader {

GroupProducer::GroupProducer(uint32_t max_batch_size)
  : batch_size_(max_batch_size),
    data_partition_num_(Options::Get().data_partitions),
    producer_() {
  batch_builders_.reserve(data_partition_num_);
  msg_builders_.reserve(data_partition_num_);
  for (uint32_t i = 0; i < data_partition_num_; i++) {
    batch_builders_.emplace_back(i);
    msg_builders_.emplace_back(Options::Get().output_kafka_topic);
    msg_builders_[i].partition(static_cast<int32_t>(Partitioner::Get().GetKafkaPartitionId(i)));
  }
}

GroupProducer::~GroupProducer() {
  FlushAll();
}

void GroupProducer::AddVertex(VertexType vtype, VertexId vid, const std::vector<AttrInfo>& attrs) {
  auto data_pid = Partitioner::Get().GetDataPartitionId(vid);
  auto& bb = batch_builders_.at(data_pid);
  bb.AddVertexUpdate(vtype, vid, attrs);
  if (bb.RecordNum() >= batch_size_) {
    Flush(data_pid);
  }
}

void GroupProducer::AddEdge(EdgeType etype, VertexType src_vtype, VertexType dst_vtype,
                            VertexId src_vid, VertexId dst_vid, const std::vector<AttrInfo>& attrs) {
  auto data_pid = Partitioner::Get().GetDataPartitionId(src_vid);
  auto& bb = batch_builders_.at(data_pid);
  bb.AddEdgeUpdate(etype, src_vtype, dst_vtype, src_vid, dst_vid, attrs);
  if (bb.RecordNum() >= batch_size_) {
    Flush(data_pid);
  }
}

void GroupProducer::FlushAll() {
  for (uint32_t i = 0; i < data_partition_num_; i++) {
    if (batch_builders_.at(i).RecordNum() > 0) {
      Flush(i);
    }
  }
}

void GroupProducer::Flush(uint32_t data_partition) {
  auto& bb = batch_builders_.at(data_partition);
  bb.Finish();
  auto& mb = msg_builders_.at(data_partition);
  mb.payload({bb.GetBufferPointer(), bb.GetBufferSize()});
  producer_.SyncProduce(mb);
  bb.Clear();
}

}  // namespace dataloader
}  // namespace dgs
