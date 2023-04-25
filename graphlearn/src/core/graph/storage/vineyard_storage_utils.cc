/* Copyright 2020-2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "core/graph/storage/vineyard_storage_utils.h"
#include "core/graph/storage/vineyard_edge_storage.h"
#include "core/graph/storage/vineyard_graph_storage.h"
#include "core/graph/storage/vineyard_node_storage.h"
#include "core/graph/storage/vineyard_topo_storage.h"

#include <memory>
#include <mutex>
#include <numeric>
#include <type_traits>

#if defined(WITH_VINEYARD)
namespace vineyard {
template class ArrowFragment<graphlearn::io::vineyard_oid_t,
                             graphlearn::io::vineyard_vid_t>;
template class ArrowVertexMap<graphlearn::io::vineyard_oid_t,
                              graphlearn::io::vineyard_vid_t>;
}
#endif

namespace graphlearn {
namespace io {

#if defined(WITH_VINEYARD)

std::shared_ptr<gl_frag_t> get_vineyard_fragment(vineyard::Client &client,
                                                 const vineyard::ObjectID object_id) {
  auto target = client.GetObject(object_id);
  std::shared_ptr<gl_frag_t> frag;
  if ((frag = std::dynamic_pointer_cast<gl_frag_t>(target)) != nullptr) {
    return frag;
  }
  std::shared_ptr<vineyard::ArrowFragmentGroup> fg;
  if ((fg = std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(target)) != nullptr) {
    // assume 1 worker per server
    for (const auto& kv : fg->Fragments()) {
      if (fg->FragmentLocations().at(kv.first) == client.instance_id()) {
        return client.GetObject<gl_frag_t>(kv.second);
      }
    }
  }
  return nullptr;
}

void init_table_accessors(const std::shared_ptr<arrow::Table>& table,
                          const std::set<std::string>& attrs,
                          std::vector<int>& i32_indexes,
                          std::vector<int>& i64_indexes,
                          std::vector<int>& f32_indexes,
                          std::vector<int>& f64_indexes,
                          std::vector<int>& s_indexes,
                          std::vector<int>& ls_indexes,
                          std::vector<const void*>& table_accessors) {
  if (table->num_rows() == 0 || table->num_columns() == 0) {
    return;
  }
  const auto& fields = table->schema()->fields();
  table_accessors.resize(fields.size(), nullptr);
  for (int idx = 0; idx < fields.size(); ++idx) {
    if (attrs.find(fields[idx]->name()) == attrs.end()) {
      continue;
    }
    auto array = table->column(idx)->chunk(0);
    table_accessors[idx] = vineyard::get_arrow_array_data(array);
    if (array->type()->Equals(arrow::int32())) {
      i32_indexes.emplace_back(idx);
    } else if (array->type()->Equals(arrow::int64())) {
      i64_indexes.emplace_back(idx);
    } else if (array->type()->Equals(arrow::float32())) {
      f32_indexes.emplace_back(idx);
    } else if (array->type()->Equals(arrow::float64())) {
      f64_indexes.emplace_back(idx);
    } else if (array->type()->Equals(arrow::utf8())) {
      s_indexes.emplace_back(idx);
    } else if (array->type()->Equals(arrow::large_utf8())) {
      ls_indexes.emplace_back(idx);
    } else {
      LOG(ERROR) << "Unsupported column type: " << array->type()->ToString();
    }
  }
}

AttributeValue *arrow_line_to_attribute_value(
                          const int row_index,
                          const std::vector<int>& i32_indexes,
                          const std::vector<int>& i64_indexes,
                          const std::vector<int>& f32_indexes,
                          const std::vector<int>& f64_indexes,
                          const std::vector<int>& s_indexes,
                          const std::vector<int>& ls_indexes,
                          const std::vector<const void*>& table_accessors) {
  auto attr = NewDataHeldAttributeValue();
  for (const auto& idx: i32_indexes) {
    auto value = vineyard::property_graph_utils::ValueGetter<int32_t>::Value(
        table_accessors[idx], row_index);
    attr->Add(static_cast<int64_t>(value));
  }
  for (const auto& idx: i64_indexes) {
    auto value = vineyard::property_graph_utils::ValueGetter<int64_t>::Value(
        table_accessors[idx], row_index);
    attr->Add(value);
  }
  for (const auto& idx: f32_indexes) {
    auto value = vineyard::property_graph_utils::ValueGetter<float>::Value(
        table_accessors[idx], row_index);
    attr->Add(value);
  }
  for (const auto& idx: f64_indexes) {
    auto value = vineyard::property_graph_utils::ValueGetter<double>::Value(
        table_accessors[idx], row_index);
    attr->Add(static_cast<float>(value));
  }
  for (const auto& idx: s_indexes) {
    auto value = std::string(
        reinterpret_cast<const arrow::StringArray*>(table_accessors[idx])->GetView(
            row_index));
    attr->Add(value);
  }
  for (const auto& idx: ls_indexes) {
    auto value = vineyard::property_graph_utils::ValueGetter<std::string>::Value(
        table_accessors[idx], row_index);
    attr->Add(value);
  }
  return attr;
}

const IndexArray get_all_in_degree(const std::shared_ptr<gl_frag_t>& frag,
                                   const label_id_t edge_label) {
  int v_label_num = frag->vertex_label_num();
  size_t total_size = 0;
  for (int label_id = 0; label_id < v_label_num; ++label_id) {
    auto id_range = frag->InnerVertices(label_id);
    total_size += id_range.size();
  }
  std::shared_ptr<IndexType> degrees(new IndexType[total_size],
                                 std::default_delete<IndexType[]>());
  IndexType* degree_ptr = degrees.get();
  size_t index = 0;
  for (int label_id = 0; label_id < v_label_num; ++label_id) {
    auto id_range = frag->InnerVertices(label_id);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      degree_ptr[index++] = frag->GetLocalInDegree(*id, edge_label);
    }
  }
  return IndexArray(degrees.get(), total_size, degrees);
}

const IndexArray get_all_out_degree(const std::shared_ptr<gl_frag_t>& frag,
                                    const label_id_t edge_label) {
  int v_label_num = frag->vertex_label_num();
  int e_label_num = frag->edge_label_num();
  size_t total_size = 0;
  for (int label_id = 0; label_id < v_label_num; ++label_id) {
    auto id_range = frag->InnerVertices(label_id);
    total_size += id_range.size();
  }
  std::shared_ptr<IndexType> degrees(new IndexType[total_size],
                                 std::default_delete<IndexType[]>());
  IndexType* degree_ptr = degrees.get();
  size_t index = 0;
  for (int label_id = 0; label_id < v_label_num; ++label_id) {
    auto id_range = frag->InnerVertices(label_id);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      degree_ptr[index++] = frag->GetLocalOutDegree(*id, edge_label);
    }
  }
  return IndexArray(degrees.get(), total_size, degrees);
}

const Array<IdType>
get_all_outgoing_neighbor_nodes(const std::shared_ptr<gl_frag_t>& frag,
                                IdType src_id, const label_id_t edge_label) {
  using nbr_unit_t = vineyard::property_graph_utils::NbrUnit<gl_frag_t::vid_t,
                                                             gl_frag_t::eid_t>;
  // FIXME extend Array to support element_size and element_offset.
  auto v = vertex_t{static_cast<uint64_t>(src_id)};
  if (!frag->IsInnerVertex(v)) {
    return Array<IdType>();
  }
  auto neighbor_list = frag->GetOutgoingAdjList(v, edge_label);
  auto gid_value_offset = frag->GetInnerVertexGid(vertex_t{0});

  std::shared_ptr<IdType> nodes(new IdType[neighbor_list.Size()],
                                std::default_delete<IdType[]>());
  IdType* node_ptr = nodes.get();
  size_t index = 0;
  for (auto iter = neighbor_list.begin_unit(); iter != neighbor_list.end_unit(); ++iter) {
#if defined(VINEYARD_USE_OID)
    node_ptr[index++] = frag->GetId(iter->get_neighbor());
#else
    node_ptr[index++] = iter->vid;
#endif
  }
  return IdArray(nodes.get(), neighbor_list.Size(), nodes);
}

const Array<IdType>
get_all_outgoing_neighbor_edges(const std::shared_ptr<gl_frag_t>& frag,
                                IdType src_id, const label_id_t edge_label) {
  using nbr_unit_t = vineyard::property_graph_utils::NbrUnit<gl_frag_t::vid_t,
                                                             gl_frag_t::eid_t>;
  // FIXME extend Array to support element_size and element_offset.
  auto v = vertex_t{static_cast<uint64_t>(src_id)};
  if (!frag->IsInnerVertex(v)) {
    return Array<IdType>();
  }
  auto neighbor_list = frag->GetOutgoingAdjList(v, edge_label);
  std::shared_ptr<IdType> edges(new IdType[neighbor_list.Size()],
                                std::default_delete<IdType[]>());
  IdType* edge_ptr = edges.get();
  size_t index = 0;
  for (auto iter = neighbor_list.begin_unit(); iter != neighbor_list.end_unit(); ++iter) {
    edge_ptr[index++] = iter->eid;
  }
  return IdArray(edges.get(), neighbor_list.Size(), edges);
}

const Array<IdType>
get_all_outgoing_neighbor_nodes(const std::shared_ptr<gl_frag_t>& frag,
                                const std::vector<IdType>& dst_lists,
                                IdType src_id, const label_id_t edge_label,
                                const std::vector<std::pair<IdType, IdType>>& edge_offsets_) {
  auto v = vertex_t{static_cast<uint64_t>(src_id)};
  if (!frag->IsInnerVertex(v)) {
    return Array<IdType>();
  }
  auto offset = edge_offsets_[frag->vertex_offset(v)];
  return IdArray(dst_lists.data() + offset.first, offset.second - offset.first);
}

const Array<IdType>
get_all_outgoing_neighbor_edges(const std::shared_ptr<gl_frag_t>& frag,
                                const std::vector<IdType>& edge_lists,
                                IdType src_id, const label_id_t edge_label,
                                const std::vector<std::pair<IdType, IdType>>& edge_offsets_) {
  auto v = vertex_t{static_cast<uint64_t>(src_id)};
  if (!frag->IsInnerVertex(v)) {
    return Array<IdType>();
  }
  auto offset = edge_offsets_[frag->vertex_offset(v)];
  std::shared_ptr<IdType> edges(new IdType[offset.second - offset.first],
                                std::default_delete<IdType[]>());
  std::iota(edges.get(), edges.get() + (offset.second - offset.first), 0);
  return IdArray(edges.get(), offset.second - offset.first, edges);
}

IdType get_edge_src_id(const std::shared_ptr<gl_frag_t>& frag,
                       const label_id_t edge_label,
                       const std::vector<IdType>& src_ids, IdType edge_id) {
#ifndef NDEBUG
  if (edge_id < src_ids.size()) {
    return src_ids[edge_id];
  } else {
    throw std::runtime_error("Not implemented since unused");
  }
#else
  return src_ids[edge_id];
#endif
}

IdType get_edge_dst_id(const std::shared_ptr<gl_frag_t>& frag,
                       const label_id_t edge_label,
                       const std::vector<IdType>& dst_ids, IdType edge_id) {
#ifndef NDEBUG
  if (edge_id < dst_ids.size()) {
    return dst_ids[edge_id];
  } else {
    throw std::runtime_error("Not implemented since unused");
  }
#else
  return dst_ids[edge_id];
#endif
}

float get_edge_weight(const std::shared_ptr<gl_frag_t>& frag,
                      const label_id_t edge_label, IdType edge_offset) {
  auto table = frag->edge_data_table(edge_label);
  int index = find_index_of_name(table->schema(), "weight");
  if (index == -1) {
    return 0.0;
  }
  auto const& array = frag->edge_data_table(edge_label)->column(index)->chunk(0);
  return static_cast<float>(
      std::dynamic_pointer_cast<arrow::DoubleArray>(array)->GetView(edge_offset));
}

int32_t get_edge_label(const std::shared_ptr<gl_frag_t>& frag,
                       const label_id_t edge_label, IdType edge_offset) {
  auto table = frag->edge_data_table(edge_label);
  int index = find_index_of_name(table->schema(), "label");
  if (index == -1) {
    return 0.0;
  }
  const auto& array = frag->edge_data_table(edge_label)->column(index)->chunk(0);
  return static_cast<int32_t>(
      std::dynamic_pointer_cast<arrow::Int64Array>(array)->GetView(edge_offset));
}

int64_t get_edge_timestamp(const std::shared_ptr<gl_frag_t>& frag,
                       const label_id_t edge_label, IdType edge_offset) {
  auto table = frag->edge_data_table(edge_label);
  int index = find_index_of_name(table->schema(), "timestamp");
  if (index == -1) {
    return  -1;
  }
  const auto& array = frag->edge_data_table(edge_label)->column(index)->chunk(0);
  return static_cast<int64_t>(
      std::dynamic_pointer_cast<arrow::Int64Array>(array)->GetView(edge_offset));
}

void init_src_dst_list(const std::shared_ptr<gl_frag_t>& frag,
                       const label_id_t edge_label,
                       const label_id_t src_node_label,
                       const label_id_t dst_node_label,
                       std::vector<IdType>& src_lists,
                       std::vector<IdType>& dst_lists,
                       std::vector<IdType>& edge_lists,
                       std::vector<std::pair<IdType, IdType>>& edge_offsets) {
  auto id_range = frag->InnerVertices(src_node_label);
  for (auto id = id_range.begin(); id < id_range.end(); ++id) {
    auto oes = frag->GetOutgoingAdjList(*id, edge_label);
#if defined(VINEYARD_USE_OID)
    auto sid = frag->GetInnerVertexId(*id);
#else
    auto sid = frag->GetInnerVertexGid(*id);
#endif
    auto e = oes.begin();
    IdType csr_begin = dst_lists.size();
    while (e != oes.end()) {
      if (frag->vertex_label(e.neighbor()) == dst_node_label) {
        break;
      }
      ++e;
    }
    while (e != oes.end()) {
      if (frag->vertex_label(e.neighbor()) != dst_node_label) {
        break;
      }
      src_lists.emplace_back(sid);
#if defined(VINEYARD_USE_OID)
      dst_lists.emplace_back(frag->GetId(e.neighbor()));
#else
      dst_lists.emplace_back(frag->Vertex2Gid(e.neighbor()));
#endif
      edge_lists.emplace_back(e.edge_id());
      ++e;
    }
    IdType csr_end = dst_lists.size();
    edge_offsets.emplace_back(csr_begin, csr_end);
  }
}

SideInfo *frag_edge_side_info(const std::shared_ptr<gl_frag_t>& frag,
                              const std::set<std::string>& attrs,
                              const std::string& edge_label_name,
                              const std::string& src_label_name,
                              const std::string& dst_label_name,
                              const label_id_t edge_label) {
  static std::map<vineyard::ObjectID,
                  std::map<std::string, std::shared_ptr<SideInfo>>>
      side_info_cache;

  static std::mutex mutex;
  std::lock_guard<std::mutex> lexical_scope_lock(mutex);
  auto cache_entry = side_info_cache[frag->id()][edge_label_name];
  if (cache_entry) {
    return cache_entry.get();
  }
  auto side_info = std::make_shared<SideInfo>();
  // compute attribute numbers of i/f/s
  auto edge_table = frag->edge_data_table(edge_label);
  auto etable_schema = edge_table->schema();
  LOG(INFO) << "etable_schema: " << etable_schema->ToString();
  const auto& fields = etable_schema->fields();
  for (size_t idx = 0; idx < fields.size(); ++idx) {
    auto field = fields[idx];
    if (attrs.find(fields[idx]->name()) == attrs.end()) {
      continue;
    }
    switch (field->type()->id()) {
    case arrow::Type::INT32:
    case arrow::Type::INT64:
      side_info->i_num += 1;
      break;
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      side_info->f_num += 1;
      break;
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
      side_info->s_num += 1;
      break;
    default:
      break;
    }
  }
  side_info->format = kDefault;
  for (auto const& field : etable_schema->fields()) {
    if (field->name() == "label") {
      side_info->format |= kLabeled;
    } else if (field->name() == "weight") {
      side_info->format |= kWeighted;
    }
    side_info->format |= kAttributed;
  }
  side_info->type = edge_label_name;
  side_info->src_type = src_label_name;
  side_info->dst_type = dst_label_name;

  // TODO: not supported
  // side_info->direction = Direction::kOrigin;

  side_info_cache[frag->id()][edge_label_name] = side_info;
  return side_info.get();
}

SideInfo *frag_node_side_info(const std::shared_ptr<gl_frag_t>& frag,
                              const std::set<std::string>& attrs,
                              const std::string& node_label_name,
                              const label_id_t node_label) {
  static std::map<vineyard::ObjectID,
                  std::map<std::string, std::shared_ptr<SideInfo>>>
      side_info_cache;

  static std::mutex mutex;
  std::lock_guard<std::mutex> lexical_scope_lock(mutex);

  auto cache_entry = side_info_cache[frag->id()][node_label_name];
  if (cache_entry) {
    return cache_entry.get();
  }
  std::cerr << "init node sideinfo " << frag->id() << std::endl;
  auto side_info = std::make_shared<SideInfo>();
  // compute attribute numbers of i/f/s
  auto node_table = frag->vertex_data_table(node_label);
  auto vtable_schema = node_table->schema();
  const auto& fields = vtable_schema->fields();
  for (size_t idx = 0; idx < fields.size(); ++idx) {
    auto field = fields[idx];
    if (attrs.find(fields[idx]->name()) == attrs.end()) {
      continue;
    }
    switch (field->type()->id()) {
    case arrow::Type::INT32:
    case arrow::Type::INT64:
      side_info->i_num += 1;
      break;
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      side_info->f_num += 1;
      break;
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
      side_info->s_num += 1;
      break;
    default:
      break;
    }
  }
  side_info->format = kDefault;
  for (size_t idx = 0; idx < fields.size(); ++idx) {
    if (fields[idx]->name() == "label") {
      side_info->format |= kLabeled;
    } else if (fields[idx]->name() == "weight") {
      side_info->format |= kWeighted;
    }
    side_info->format |= kAttributed;
  }
  side_info->type = node_label_name;
  side_info_cache[frag->id()][node_label_name] = side_info;
  return side_info.get();
}

int64_t find_index_of_name(const std::shared_ptr<arrow::Schema>& schema,
                           const std::string& name) {
  for (int64_t index = 0; index < schema->num_fields(); ++index) {
    if (schema->field(index)->name() == name) {
      return index;
    }
  }
  return -1;
}

void ArrowRefAttributeValue::FillInts(Tensor *tensor) const {
  for (const auto& idx: i32_indexes_) {
    auto value = vineyard::property_graph_utils::ValueGetter<int32_t>::Value(
        table_accessors_[idx], row_index_);
    tensor->AddInt64(static_cast<int64_t>(value));
  }
  for (const auto& idx: i64_indexes_) {
    auto value = vineyard::property_graph_utils::ValueGetter<int64_t>::Value(
        table_accessors_[idx], row_index_);
    tensor->AddInt64(value);
  }
}

void ArrowRefAttributeValue::FillFloats(Tensor *tensor) const {
  for (const auto& idx: f32_indexes_) {
    auto value = vineyard::property_graph_utils::ValueGetter<float>::Value(
        table_accessors_[idx], row_index_);
    tensor->AddFloat(value);
  }
  for (const auto& idx: f64_indexes_) {
    auto value = vineyard::property_graph_utils::ValueGetter<double>::Value(
        table_accessors_[idx], row_index_);
    tensor->AddFloat(static_cast<float>(value));
  }
}

void ArrowRefAttributeValue::FillStrings(Tensor *tensor) const {
  for (const auto& idx: s_indexes_) {
    auto value = std::string(
        reinterpret_cast<const arrow::StringArray*>(table_accessors_[idx])->GetView(
            row_index_));
    tensor->AddString(value);
  }
  for (const auto& idx: ls_indexes_) {
    auto value = vineyard::property_graph_utils::ValueGetter<std::string>::Value(
        table_accessors_[idx], row_index_);
    tensor->AddString(value);
  }
}

#endif

GraphStorage *NewVineyardGraphStorage(const std::string& edge_type,
    const std::string& view_type,
    const std::string& use_attrs) {
#if defined(WITH_VINEYARD)
  LOG(INFO) << "create vineyard graph storage";
#if defined(VINEYARD_USE_OID)
  LOG(INFO) << "use external ID as node id";
#endif
  return new VineyardGraphStorage(edge_type, view_type, use_attrs);
#else
  throw std::runtime_error("create graph stroage: vineyard is not enabled");
#endif
}

NodeStorage *NewVineyardNodeStorage(const std::string& node_type,
    const std::string& view_type,
    const std::string& use_attrs) {
#if defined(WITH_VINEYARD)
  LOG(INFO) << "create vineyard node storage";
#if defined(VINEYARD_USE_OID)
  LOG(INFO) << "use external ID as node id";
#endif
  return new VineyardNodeStorage(node_type, view_type, use_attrs);
#else
  throw std::runtime_error("create node storage: vineyard is not enabled");
#endif
}

} // namespace io
} // namespace graphlearn
