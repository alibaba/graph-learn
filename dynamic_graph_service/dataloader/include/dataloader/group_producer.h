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

#ifndef FILE_LOADER_GROUP_PRODUCER_H_
#define FILE_LOADER_GROUP_PRODUCER_H_

#include "dataloader/batch_builder.h"
#include "dataloader/batch_producer.h"

namespace dgs {
namespace dataloader {

class GroupProducer {
public:
  explicit GroupProducer(uint32_t max_batch_size);
  ~GroupProducer();

  void AddVertex(VertexType vtype, VertexId vid, const std::vector<AttrInfo>& attrs);
  void AddEdge(EdgeType etype, VertexType src_vtype, VertexType dst_vtype,
               VertexId src_vid, VertexId dst_vid, const std::vector<AttrInfo>& attrs);

  void FlushAll();

private:
  void Flush(uint32_t data_partition);

private:
  const uint32_t batch_size_;
  const uint32_t data_partition_num_;
  BatchProducer producer_;
  std::vector<BatchBuilder> batch_builders_;
  std::vector<cppkafka::MessageBuilder> msg_builders_;
};

}  // namespace dataloader
}  // namespace dgs


#endif // FILE_LOADER_GROUP_PRODUCER_H_
