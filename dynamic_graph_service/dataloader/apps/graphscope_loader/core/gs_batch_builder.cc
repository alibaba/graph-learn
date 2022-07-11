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

#include "gs_batch_builder.h"

namespace dgs {
namespace dataloader {
namespace gs {

void GSBatchBuilder::AddVertexUpdate(const lgraph::log_subscription::VertexInsertInfo& info) {
  auto flat_vtype = static_cast<VertexType>(info.GetLabelId());
  VertexId flat_vid = info.GetVertexId();
  auto& prop_map = info.GetPropMap();
  auto flat_attrs = AddAttributes(prop_map);
  auto flat_vertex = CreateVertexRecordRep(
      builder_, flat_vtype, flat_vid, flat_attrs);
  auto flat_record = CreateRecordRep(
      builder_, RecordUnionRep_VertexRecordRep, flat_vertex.Union());
  records_.emplace_back(flat_record);
}

void GSBatchBuilder::AddEdgeUpdate(const lgraph::log_subscription::EdgeInsertInfo& info) {
  auto& edge_rel = info.GetEdgeRelation();
  auto flat_etype = static_cast<EdgeType>(edge_rel.edge_label_id);
  auto flat_src_vtype = static_cast<VertexType>(edge_rel.src_vertex_label_id);
  auto flat_dst_vtype = static_cast<VertexType>(edge_rel.dst_vertex_label_id);
  auto& edge_id = info.GetEdgeId();
  VertexId flat_src_id = edge_id.src_vertex_id;
  VertexId flat_dst_id = edge_id.dst_vertex_id;
  auto& prop_map = info.GetPropMap();
  auto flat_attrs = AddAttributes(prop_map);
  auto flat_edge = CreateEdgeRecordRep(
      builder_, flat_etype, flat_src_vtype, flat_dst_vtype, flat_src_id, flat_dst_id, flat_attrs);
  auto flat_record = CreateRecordRep(
      builder_, RecordUnionRep_EdgeRecordRep, flat_edge.Union());
  records_.emplace_back(flat_record);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<AttributeRecordRep>>>
GSBatchBuilder::AddAttributes(
    const std::unordered_map<lgraph::PropertyId, lgraph::log_subscription::PropertyInfo>& prop_map) {
  std::vector<flatbuffers::Offset<AttributeRecordRep>> flat_attrs_vec;
  flat_attrs_vec.reserve(prop_map.size());
  for (auto& entry : prop_map) {
    auto flat_attr_type = static_cast<AttributeType>(entry.first);
    auto flat_value_type = static_cast<AttributeValueTypeRep>(entry.second.GetDataType());
    auto& value_bytes = entry.second.GetValueBytes();
    auto flat_value_bytes = AddAttributeValueBytes(value_bytes, entry.second.GetDataType());
    auto flat_attr = CreateAttributeRecordRep(
        builder_, flat_attr_type, flat_value_type, flat_value_bytes);
    flat_attrs_vec.push_back(flat_attr);
  }
  return builder_.CreateVectorOfSortedTables(&flat_attrs_vec);
}

flatbuffers::Offset<flatbuffers::Vector<int8_t>>
GSBatchBuilder::AddAttributeValueBytes(const std::string& bytes, const lgraph::DataType& prop_value_type) {
  if (prop_value_type == lgraph::INT ||
  prop_value_type == lgraph::LONG ||
  prop_value_type == lgraph::FLOAT ||
  prop_value_type == lgraph::DOUBLE) {
    std::string reversed_bytes;
    reversed_bytes.resize(bytes.size());
    std::reverse_copy(bytes.begin(), bytes.end(), reversed_bytes.begin());
    return builder_.CreateVector(reinterpret_cast<const int8_t*>(reversed_bytes.data()), reversed_bytes.size());
  } else if (prop_value_type == lgraph::STRING) {
    return builder_.CreateVector(reinterpret_cast<const int8_t*>(bytes.data()), bytes.size());
  } else {
    std::cout << "[Warning] Got unsupported attribute value type!\n";
    return builder_.CreateVector(static_cast<const int8_t*>(nullptr), 0);
  }
}

void GSBatchBuilder::AddVertexUpdate(lgraph::db::Vertex* vertex, const lgraph::Schema& store_schema) {
  auto flat_vtype = static_cast<VertexType>(vertex->GetLabelId());
  VertexId flat_vid = vertex->GetVertexId();
  auto prop_iter = vertex->GetPropertyIterator();
  auto flat_attrs = AddAttributes(&prop_iter, store_schema);
  auto flat_vertex = CreateVertexRecordRep(
      builder_, flat_vtype, flat_vid, flat_attrs);
  auto flat_record = CreateRecordRep(
      builder_, RecordUnionRep_VertexRecordRep, flat_vertex.Union());
  records_.emplace_back(flat_record);
}

void GSBatchBuilder::AddEdgeUpdate(lgraph::db::Edge* edge, const lgraph::Schema& store_schema) {
  auto edge_rel = edge->GetEdgeRelation();
  auto flat_etype = static_cast<EdgeType>(edge_rel.edge_label_id);
  auto flat_src_vtype = static_cast<VertexType>(edge_rel.src_vertex_label_id);
  auto flat_dst_vtype = static_cast<VertexType>(edge_rel.dst_vertex_label_id);
  auto edge_id = edge->GetEdgeId();
  VertexId flat_src_id = edge_id.src_vertex_id;
  VertexId flat_dst_id = edge_id.dst_vertex_id;
  auto prop_iter = edge->GetPropertyIterator();
  auto flat_attrs = AddAttributes(&prop_iter, store_schema);
  auto flat_edge = CreateEdgeRecordRep(
      builder_, flat_etype, flat_src_vtype, flat_dst_vtype,flat_src_id, flat_dst_id, flat_attrs);
  auto flat_record = CreateRecordRep(
      builder_, RecordUnionRep_EdgeRecordRep, flat_edge.Union());
  records_.emplace_back(flat_record);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<AttributeRecordRep>>>
GSBatchBuilder::AddAttributes(lgraph::db::PropertyIterator* prop_iter, const lgraph::Schema& store_schema) {
  std::vector<flatbuffers::Offset<AttributeRecordRep>> flat_attrs_vec;
  while (true) {
    auto p = prop_iter->Next().unwrap();
    if (!p.Valid()) {
      break;
    }
    auto prop_id = p.GetPropertyId();
    auto flat_attr_type = static_cast<AttributeType>(prop_id);
    auto prop_value_type = store_schema.GetPropDef(prop_id).GetDataType();
    auto flat_value_type = static_cast<AttributeValueTypeRep>(prop_value_type);
    auto flat_value_bytes = AddAttributeValueBytes(&p, prop_value_type);
    auto flat_attr = CreateAttributeRecordRep(
        builder_, flat_attr_type, flat_value_type, flat_value_bytes);
    flat_attrs_vec.push_back(flat_attr);
  }
  return builder_.CreateVectorOfSortedTables(&flat_attrs_vec);
}

flatbuffers::Offset<flatbuffers::Vector<int8_t>>
GSBatchBuilder::AddAttributeValueBytes(lgraph::db::Property* prop, const lgraph::DataType& prop_value_type) {
  flatbuffers::Offset<flatbuffers::Vector<int8_t>> flat_value_bytes;
  switch (prop_value_type) {
    case lgraph::INT: {
      auto value = prop->GetAsInt32().unwrap();
      flat_value_bytes = builder_.CreateVector(reinterpret_cast<int8_t*>(&value), sizeof(int32_t));
      break;
    }
    case lgraph::LONG: {
      auto value = prop->GetAsInt64().unwrap();
      flat_value_bytes = builder_.CreateVector(reinterpret_cast<int8_t*>(&value), sizeof(int64_t));
      break;
    }
    case lgraph::FLOAT: {
      auto value = prop->GetAsFloat().unwrap();
      flat_value_bytes = builder_.CreateVector(reinterpret_cast<int8_t*>(&value), sizeof(float));
      break;
    }
    case lgraph::DOUBLE: {
      auto value = prop->GetAsDouble().unwrap();
      flat_value_bytes = builder_.CreateVector(reinterpret_cast<int8_t*>(&value), sizeof(double));
      break;
    }
    case lgraph::STRING: {
      auto str_slice = prop->GetAsStr().unwrap();
      flat_value_bytes = builder_.CreateVector(static_cast<int8_t*>(str_slice.data), str_slice.len);
      break;
    }
    default: {
      std::cout << "[Warning] Got unsupported attribute value type!\n";
      flat_value_bytes = builder_.CreateVector(static_cast<int8_t*>(nullptr), 0);
    }
  }
  return flat_value_bytes;
}

}  // namespace gs
}  // namespace dataloader
}  // namespace dgs
