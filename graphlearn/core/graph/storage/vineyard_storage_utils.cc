#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"
#include "graphlearn/core/graph/storage/vineyard_edge_storage.h"
#include "graphlearn/core/graph/storage/vineyard_graph_storage.h"
#include "graphlearn/core/graph/storage/vineyard_node_storage.h"
#include "graphlearn/core/graph/storage/vineyard_topo_storage.h"

#include <memory>
#include <mutex>
#include <type_traits>

#if defined(WITH_VINEYARD)
namespace vineyard {
template class ArrowFragment<graphlearn::io::vineyard_oid_t,
                             graphlearn::io::vineyard_vid_t>;
}
#endif

namespace graphlearn {
namespace io {

#if defined(WITH_VINEYARD)

void init_table_accessors(std::shared_ptr<arrow::Table> const &table,
                          std::set<std::string> const &attrs,
                          std::vector<int> &i32_indexes,
                          std::vector<int> &i64_indexes,
                          std::vector<int> &f32_indexes,
                          std::vector<int> &f64_indexes,
                          std::vector<int> &s_indexes,
                          std::vector<int> &ls_indexes,
                          std::vector<const void*> &table_accessors) {
  if (table->num_rows() == 0 || table->num_columns() == 0) {
    return;
  }
  auto const &fields = table->schema()->fields();
  table_accessors.resize(fields.size(), nullptr);
  for (int idx = 0; idx < fields.size(); ++idx) {
    if (attrs.find(fields[idx]->name()) == attrs.end()) {
      continue;
    }
    auto array = table->column(idx)->chunk(0);
    table_accessors[idx] = vineyard::get_arrow_array_ptr(array);
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
                          const std::vector<int> &i32_indexes,
                          const std::vector<int> &i64_indexes,
                          const std::vector<int> &f32_indexes,
                          const std::vector<int> &f64_indexes,
                          const std::vector<int> &s_indexes,
                          const std::vector<int> &ls_indexes,
                          const std::vector<const void*> &table_accessors) {
  auto attr = NewDataHeldAttributeValue();
  for (auto const &idx: i32_indexes) {
    auto value = vineyard::property_graph_utils::ValueGetter<int32_t>::Value(
        table_accessors[idx], row_index);
    attr->Add(static_cast<int64_t>(value));
  }
  for (auto const &idx: i64_indexes) {
    auto value = vineyard::property_graph_utils::ValueGetter<int64_t>::Value(
        table_accessors[idx], row_index);
    attr->Add(value);
  }
  for (auto const &idx: f32_indexes) {
    auto value = vineyard::property_graph_utils::ValueGetter<float>::Value(
        table_accessors[idx], row_index);
    attr->Add(value);
  }
  for (auto const &idx: f64_indexes) {
    auto value = vineyard::property_graph_utils::ValueGetter<double>::Value(
        table_accessors[idx], row_index);
    attr->Add(static_cast<float>(value));
  }
  for (auto const &idx: s_indexes) {
    auto value = std::string(
        reinterpret_cast<const arrow::StringArray*>(table_accessors[idx])->GetView(
            row_index));
    attr->Add(value);
  }
  for (auto const &idx: ls_indexes) {
    auto value = vineyard::property_graph_utils::ValueGetter<std::string>::Value(
        table_accessors[idx], row_index);
    attr->Add(value);
  }
  return attr;
}

const IndexList *get_all_in_degree(std::shared_ptr<gl_frag_t> const &frag,
                                   const label_id_t edge_label) {
  int v_label_num = frag->vertex_label_num();
  auto degree_list = new IndexList();
  for (int label_id = 0; label_id < v_label_num; ++label_id) {
    auto id_range = frag->InnerVertices(label_id);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      auto degree = frag->GetLocalInDegree(*id, edge_label);
      if (degree > 0) {
        degree_list->emplace_back(degree);
      }
    }
  }
  return degree_list;
}

const IndexList *get_all_out_degree(std::shared_ptr<gl_frag_t> const &frag,
                                    const label_id_t edge_label) {
  int v_label_num = frag->vertex_label_num();
  int e_label_num = frag->edge_label_num();
  auto degree_list = new IndexList();
  for (int label_id = 0; label_id < v_label_num; ++label_id) {
    auto id_range = frag->InnerVertices(label_id);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      auto degree = frag->GetLocalOutDegree(*id, edge_label);
      if (degree > 0) {
        degree_list->emplace_back(degree);
      }
    }
  }
  return degree_list;
}

const Array<IdType>
get_all_outgoing_neighbor_nodes(std::shared_ptr<gl_frag_t> const &frag,
                                IdType src_id, const label_id_t edge_label) {
  using nbr_unit_t = vineyard::property_graph_utils::NbrUnit<gl_frag_t::vid_t,
                                                             gl_frag_t::eid_t>;
  // FIXME extend Array to support element_size and element_offset.
  auto v = vertex_t{static_cast<uint64_t>(src_id)};
  if (!frag->IsInnerVertex(v)) {
    return Array<IdType>();
  }
  std::vector<const IdType *> values;
  std::vector<int32_t> sizes;
  auto neighbor_list = frag->GetOutgoingAdjList(v, edge_label);
  auto gid_value_offset = frag->GetInnerVertexGid(vertex_t{0});
  values.emplace_back(
      reinterpret_cast<const IdType *>(neighbor_list.begin_unit()));
  sizes.emplace_back(neighbor_list.Size());
  return Array<IdType>(std::make_shared<MultiArray<IdType>>(
      values, sizes, sizeof(nbr_unit_t), __builtin_offsetof(nbr_unit_t, vid), gid_value_offset));
}

const Array<IdType>
get_all_outgoing_neighbor_edges(std::shared_ptr<gl_frag_t> const &frag,
                                IdType src_id, const label_id_t edge_label) {
  using nbr_unit_t = vineyard::property_graph_utils::NbrUnit<gl_frag_t::vid_t,
                                                             gl_frag_t::eid_t>;
  // FIXME extend Array to support element_size and element_offset.
  auto v = vertex_t{static_cast<uint64_t>(src_id)};
  if (!frag->IsInnerVertex(v)) {
    return Array<IdType>();
  }
  std::vector<const IdType *> values;
  std::vector<int32_t> sizes;
  auto neighbor_list = frag->GetOutgoingAdjList(v, edge_label);
  values.emplace_back(
      reinterpret_cast<const IdType *>(neighbor_list.begin_unit()));
  sizes.emplace_back(neighbor_list.Size());
  return Array<IdType>(std::make_shared<MultiArray<IdType>>(
      values, sizes, sizeof(nbr_unit_t), __builtin_offsetof(nbr_unit_t, eid)));
}

const Array<IdType>
get_all_outgoing_neighbor_nodes(std::shared_ptr<gl_frag_t> const &frag,
                                std::vector<IdType> const &dst_lists,
                                IdType src_id, const label_id_t edge_label,
                                std::vector<std::pair<IdType, IdType>> const &edge_offsets_) {
  auto v = vertex_t{static_cast<uint64_t>(src_id)};
  if (!frag->IsInnerVertex(v)) {
    return Array<IdType>();
  }
  auto offset = edge_offsets_[frag->vertex_offset(v)];
  return IdArray(dst_lists.data() + offset.first, offset.second - offset.first);
}

const Array<IdType>
get_all_outgoing_neighbor_edges(std::shared_ptr<gl_frag_t> const &frag,
                                std::vector<IdType> const &edge_lists,
                                IdType src_id, const label_id_t edge_label,
                                std::vector<std::pair<IdType, IdType>> const &edge_offsets_) {
  auto v = vertex_t{static_cast<uint64_t>(src_id)};
  if (!frag->IsInnerVertex(v)) {
    return Array<IdType>();
  }
  auto offset = edge_offsets_[frag->vertex_offset(v)];
  return IdArray(offset.first, offset.second);
}

IdType get_edge_src_id(std::shared_ptr<gl_frag_t> const &frag,
                       label_id_t const edge_label,
                       std::vector<IdType> const &src_ids, IdType edge_id) {
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

IdType get_edge_dst_id(std::shared_ptr<gl_frag_t> const &frag,
                       label_id_t const edge_label,
                       std::vector<IdType> const &dst_ids, IdType edge_id) {
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

float get_edge_weight(std::shared_ptr<gl_frag_t> const &frag,
                      label_id_t const edge_label, IdType edge_offset) {
  auto table = frag->edge_data_table(edge_label);
  int index = find_index_of_name(table->schema(), "weight");
  if (index == -1) {
    return 0.0;
  }
  auto const &array = frag->edge_data_table(edge_label)->column(index)->chunk(0);
  return static_cast<float>(
      std::dynamic_pointer_cast<arrow::DoubleArray>(array)->GetView(edge_offset));
}

int32_t get_edge_label(std::shared_ptr<gl_frag_t> const &frag,
                       label_id_t const edge_label, IdType edge_offset) {
  auto table = frag->edge_data_table(edge_label);
  int index = find_index_of_name(table->schema(), "label");
  if (index == -1) {
    return 0.0;
  }
  auto const &array = frag->edge_data_table(edge_label)->column(index)->chunk(0);
  return static_cast<int32_t>(
      std::dynamic_pointer_cast<arrow::Int64Array>(array)->GetView(edge_offset));
}

void init_src_dst_list(std::shared_ptr<gl_frag_t> const &frag,
                       label_id_t const edge_label,
                       label_id_t const src_node_label,
                       label_id_t const dst_node_label,
                       std::vector<IdType> &src_lists,
                       std::vector<IdType> &dst_lists,
                       std::vector<IdType> &edge_lists,
                       std::vector<std::pair<IdType, IdType>> &edge_offsets) {
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

SideInfo *frag_edge_side_info(std::shared_ptr<gl_frag_t> const &frag,
                              std::set<std::string> const &attrs,
                              std::string const &edge_label_name,
                              std::string const &src_label_name,
                              std::string const &dst_label_name,
                              label_id_t const edge_label) {
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
  auto const &fields = etable_schema->fields();
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
  for (auto const &field : etable_schema->fields()) {
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

SideInfo *frag_node_side_info(std::shared_ptr<gl_frag_t> const &frag,
                              std::set<std::string> const &attrs,
                              std::string const &node_label_name,
                              label_id_t const node_label) {
  static std::map<vineyard::ObjectID,
                  std::map<std::string, std::shared_ptr<SideInfo>>>
      side_info_cache;

  static std::mutex mutex;
  std::lock_guard<std::mutex> lexical_scope_lock(mutex);

  auto cache_entry = side_info_cache[frag->id()][node_label_name];
  if (cache_entry) {
    return cache_entry.get();
  }
  std::cerr << "init node sideinfo " << frag->id();
  auto side_info = std::make_shared<SideInfo>();
  // compute attribute numbers of i/f/s
  auto node_table = frag->vertex_data_table(node_label);
  auto vtable_schema = node_table->schema();
  auto const &fields = vtable_schema->fields();
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

int64_t find_index_of_name(std::shared_ptr<arrow::Schema> const &schema,
                           std::string const &name) {
  for (int64_t index = 0; index < schema->num_fields(); ++index) {
    if (schema->field(index)->name() == name) {
      return index;
    }
  }
  return -1;
}

void ArrowRefAttributeValue::FillInts(Tensor *tensor) const {
  for (auto const &idx: i32_indexes_) {
    auto value = vineyard::property_graph_utils::ValueGetter<int32_t>::Value(
        table_accessors_[idx], row_index_);
    tensor->AddInt64(static_cast<int64_t>(value));
  }
  for (auto const &idx: i64_indexes_) {
    auto value = vineyard::property_graph_utils::ValueGetter<int64_t>::Value(
        table_accessors_[idx], row_index_);
    tensor->AddInt64(value);
  }
}

void ArrowRefAttributeValue::FillFloats(Tensor *tensor) const {
  for (auto const &idx: f32_indexes_) {
    auto value = vineyard::property_graph_utils::ValueGetter<float>::Value(
        table_accessors_[idx], row_index_);
    tensor->AddFloat(value);
  }
  for (auto const &idx: f64_indexes_) {
    auto value = vineyard::property_graph_utils::ValueGetter<double>::Value(
        table_accessors_[idx], row_index_);
    tensor->AddFloat(static_cast<float>(value));
  }
}

void ArrowRefAttributeValue::FillStrings(Tensor *tensor) const {
  for (auto const &idx: s_indexes_) {
    auto value = std::string(
        reinterpret_cast<const arrow::StringArray*>(table_accessors_[idx])->GetView(
            row_index_));
    tensor->AddString(value);
  }
  for (auto const &idx: ls_indexes_) {
    auto value = vineyard::property_graph_utils::ValueGetter<std::string>::Value(
        table_accessors_[idx], row_index_);
    tensor->AddString(value);
  }
}

#endif

GraphStorage *NewVineyardGraphStorage(const std::string &edge_type,
    const std::string &view_type,
    const std::string &use_attrs) {
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

NodeStorage *NewVineyardNodeStorage(const std::string &node_type,
    const std::string &view_type,
    const std::string &use_attrs) {
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
