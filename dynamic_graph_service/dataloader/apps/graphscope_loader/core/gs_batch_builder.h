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

#ifndef GRAPHSCOPE_LOADER_GS_BATCH_BUILDER_H_
#define GRAPHSCOPE_LOADER_GS_BATCH_BUILDER_H_

#include "lgraph/db/readonly_db.h"
#include "lgraph/log_subscription/operation.h"

#include "dataloader/batch_builder.h"

namespace dgs {
namespace dataloader {
namespace gs {

class GSBatchBuilder : public BatchBuilder {
public:
  explicit GSBatchBuilder(PartitionId data_partition) : BatchBuilder(data_partition) {}
  ~GSBatchBuilder() = default;

  GSBatchBuilder(const GSBatchBuilder&) = delete;
  GSBatchBuilder& operator=(const GSBatchBuilder&) = delete;
  GSBatchBuilder(GSBatchBuilder&&) = default;
  GSBatchBuilder& operator=(GSBatchBuilder&&) = default;

  /// Add vertex/edge update from subscribing logs into builder.
  ///
  /// \param info manage the record infos in the structure supported by maxgraph sdk.
  void AddVertexUpdate(const lgraph::log_subscription::VertexInsertInfo& info);
  void AddEdgeUpdate(const lgraph::log_subscription::EdgeInsertInfo& info);

  /// Add vertex/edge update from db records into builder.
  ///
  /// \param vertex/edge manage the record infos in the structure supported by backup store.
  /// \param store_schema the graph schema of backup store.
  void AddVertexUpdate(lgraph::db::Vertex* vertex, const lgraph::Schema& store_schema);
  void AddEdgeUpdate(lgraph::db::Edge* edge, const lgraph::Schema& store_schema);

private:
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<AttributeRecordRep>>>
  AddAttributes(const std::unordered_map<lgraph::PropertyId, lgraph::log_subscription::PropertyInfo>& prop_map);

  flatbuffers::Offset<flatbuffers::Vector<int8_t>>
  AddAttributeValueBytes(const std::string& bytes, const lgraph::DataType& prop_value_type);

  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<AttributeRecordRep>>>
  AddAttributes(lgraph::db::PropertyIterator* prop_iter, const lgraph::Schema& store_schema);

  flatbuffers::Offset<flatbuffers::Vector<int8_t>>
  AddAttributeValueBytes(lgraph::db::Property* prop, const lgraph::DataType& prop_value_type);
};

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs

#endif // GRAPHSCOPE_LOADER_GS_BATCH_BUILDER_H_
