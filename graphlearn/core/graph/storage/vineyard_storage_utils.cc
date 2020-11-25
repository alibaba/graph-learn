#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"
#include "graphlearn/core/graph/storage/vineyard_edge_storage.h"
#include "graphlearn/core/graph/storage/vineyard_graph_storage.h"
#include "graphlearn/core/graph/storage/vineyard_node_storage.h"
#include "graphlearn/core/graph/storage/vineyard_topo_storage.h"

#include <memory>
#include <mutex>

#if defined(WITH_VINEYARD)
namespace vineyard {
template class ArrowFragment<graphlearn::io::vineyard_oid_t,
                             graphlearn::io::vineyard_vid_t>;
}
#endif

namespace graphlearn {
namespace io {

#if defined(WITH_VINEYARD)

AttributeValue *
arrow_line_to_attribute_value(std::shared_ptr<arrow::Table> table,
                              int row_index, int start_index) {
  auto attr = NewDataHeldAttributeValue();
  VINEYARD_ASSERT(row_index < table->num_rows());
  // NOTE: the last column is id column, cast off
  for (int idx = start_index; idx < table->num_columns() - 1; ++idx) {
    auto arr = table->column(idx)->chunk(0);
    switch (arr->type()->id()) {
    case arrow::Type::INT64: {
      int64_t value =
          std::dynamic_pointer_cast<
              typename vineyard::ConvertToArrowType<int64_t>::ArrayType>(arr)
              ->Value(row_index);
#ifndef NDEBUG
      std::cerr << "int64 value: " << value << std::endl;
#endif
      attr->Add(value);
    } break;
    case arrow::Type::FLOAT: {
      float value =
          std::dynamic_pointer_cast<
              typename vineyard::ConvertToArrowType<float>::ArrayType>(arr)
              ->Value(row_index);
#ifndef NDEBUG
      std::cerr << "float value: " << value << std::endl;
#endif
      attr->Add(value);
    } break;
    case arrow::Type::DOUBLE: {
      double value =
          std::dynamic_pointer_cast<
              typename vineyard::ConvertToArrowType<double>::ArrayType>(arr)
              ->Value(row_index);
#ifndef NDEBUG
      std::cerr << "double value: " << value << std::endl;
#endif
      attr->Add(static_cast<float>(value));
    } break;
    case arrow::Type::STRING: {
      std::string value =
          std::dynamic_pointer_cast<
              typename vineyard::ConvertToArrowType<std::string>::ArrayType>(
              arr)
              ->GetString(row_index);
#ifndef NDEBUG
      std::cerr << "std::string value: " << value << std::endl;
#endif
      attr->Add(value);
    } break;
    default:
      std::cerr << "Unsupported attribute type: " << arr->type()->ToString()
                << std::endl;
    }
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
      auto degree = frag->GetLocalInDegree(id, edge_label);
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
      auto degree = frag->GetLocalOutDegree(id, edge_label);
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
  std::vector<const IdType *> values;
  std::vector<int32_t> sizes;
  auto neighbor_list = frag->GetOutgoingAdjList(
      vertex_t{static_cast<uint64_t>(src_id)}, edge_label);
  values.emplace_back(
      reinterpret_cast<const IdType *>(neighbor_list.begin_unit()));
  sizes.emplace_back(neighbor_list.Size());
  return Array<IdType>(std::make_shared<MultiArray<IdType>>(
      values, sizes, sizeof(nbr_unit_t), __builtin_offsetof(nbr_unit_t, vid)));
}

const Array<IdType>
get_all_outgoing_neighbor_edges(std::shared_ptr<gl_frag_t> const &frag,
                                IdType src_id, const label_id_t edge_label) {
  using nbr_unit_t = vineyard::property_graph_utils::NbrUnit<gl_frag_t::vid_t,
                                                             gl_frag_t::eid_t>;
  // FIXME extend Array to support element_size and element_offset.
  std::vector<const IdType *> values;
  std::vector<int32_t> sizes;
  auto neighbor_list = frag->GetOutgoingAdjList(
      vertex_t{static_cast<uint64_t>(src_id)}, edge_label);
  values.emplace_back(
      reinterpret_cast<const IdType *>(neighbor_list.begin_unit()));
  sizes.emplace_back(neighbor_list.Size());
  return Array<IdType>(std::make_shared<MultiArray<IdType>>(
      values, sizes, sizeof(nbr_unit_t), __builtin_offsetof(nbr_unit_t, eid)));
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
                      label_id_t const edge_label, IdType edge_id) {
  auto table = frag->edge_data_table(edge_label);
  int index = find_index_of_name(table->schema(), "weight");
  if (index == -1) {
    return 0.0;
  }
  auto const &array = frag->edge_data_table(edge_label)->column(index)->chunk(0);
  return static_cast<float>(
      std::dynamic_pointer_cast<arrow::DoubleArray>(array)->GetView(edge_id));
}

int32_t get_edge_label(std::shared_ptr<gl_frag_t> const &frag,
                       label_id_t const edge_label, IdType edge_id) {
  auto table = frag->edge_data_table(edge_label);
  int index = find_index_of_name(table->schema(), "label");
  if (index == -1) {
    return 0.0;
  }
  auto const &array = frag->edge_data_table(edge_label)->column(index)->chunk(0);
  return static_cast<int32_t>(
      std::dynamic_pointer_cast<arrow::Int64Array>(array)->GetView(edge_id));
}

Attribute get_edge_attribute(std::shared_ptr<gl_frag_t> const &frag,
                             label_id_t const edge_label, IdType edge_id) {
  auto table = frag->edge_data_table(edge_label);
  return Attribute(arrow_line_to_attribute_value(table, edge_id, 0), true);
}

void initSrcDstList(std::shared_ptr<gl_frag_t> const &frag,
                    label_id_t const edge_label, std::vector<IdType> &src_lists,
                    std::vector<IdType> &dst_lists) {
  src_lists.resize(frag->edge_data_table(edge_label)->num_rows());
  dst_lists.resize(frag->edge_data_table(edge_label)->num_rows());
  for (label_id_t node_label = 0; node_label < frag->vertex_label_num();
       ++node_label) {
    auto id_range = frag->InnerVertices(node_label);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      auto oes = frag->GetOutgoingAdjList(id, edge_label);
      for (auto &e : oes) {
        src_lists[e.edge_id()] = frag->GetInnerVertexGid(id);
        dst_lists[e.edge_id()] = frag->Vertex2Gid(e.neighbor());
      }
    }
  }
}

SideInfo *frag_edge_side_info(std::shared_ptr<gl_frag_t> const &frag,
                              label_id_t const edge_label) {
  static std::map<vineyard::ObjectID,
                  std::map<label_id_t, std::shared_ptr<SideInfo>>>
      side_info_cache;

  static std::mutex mutex;
  std::lock_guard<std::mutex> lexical_scope_lock(mutex);
  auto cache_entry = side_info_cache[frag->id()][edge_label];
  if (cache_entry) {
    return cache_entry.get();
  }
  auto side_info = std::make_shared<SideInfo>();
  // compute attribute numbers of i/f/s
  auto edge_table = frag->edge_data_table(edge_label);
  auto etable_schema = edge_table->schema();
  LOG(INFO) << "etable_schema: " << etable_schema->ToString();
  for (size_t idx = 0; idx < etable_schema->fields().size(); ++idx) {
    auto field = etable_schema->fields()[idx];
    switch (field->type()->id()) {
    case arrow::Type::INT64:
      side_info->i_num += 1;
      LOG(INFO) << "matched a int";
      break;
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      side_info->f_num += 1;
      LOG(INFO) << "matched a float";
      break;
    case arrow::Type::STRING:
      side_info->s_num += 1;
      LOG(INFO) << "matched a string";
      break;
    default:
      break;
    }
  }
  side_info->format = kDefault;
  for (auto const &field : etable_schema->fields()) {
    // if (field->name() == "label") {
    //   side_info->format |= kLabeled;
    // } else if (field->name() == "weight") {
    //   side_info->format |= kWeighted;
    // } else {
    //   // otherwise we have attributes
    //   side_info->format |= kAttributed;
    // }
    side_info->format |= kAttributed;
  }
  side_info->type = std::to_string(edge_label);
  // TODO: in vineyard's data model, edges of the same label can have arbitary
  // kinds of sources and destinations.
  //
  // Thus we just inspect the label of first src/dst
  // gl_frag_t::vid_t first_src_id = frag->edge_srcs(edge_label)->Value(0);
  // gl_frag_t::vid_t first_dst_id = frag->edge_dsts(edge_label)->Value(0);
  // side_info->src_type = std::to_string(
  //     frag->vertex_label(vertex_t{first_src_id}));
  // side_info->dst_type = std::to_string(
  //     frag->vertex_label(vertex_t{first_dst_id}));
  // side_info->direction = Direction::kOrigin;

  side_info_cache[frag->id()][edge_label] = side_info;
  return side_info.get();
}

SideInfo *frag_node_side_info(std::shared_ptr<gl_frag_t> const &frag,
                              label_id_t const node_label) {
  static std::map<vineyard::ObjectID,
                  std::map<label_id_t, std::shared_ptr<SideInfo>>>
      side_info_cache;

  static std::mutex mutex;
  std::lock_guard<std::mutex> lexical_scope_lock(mutex);

  auto cache_entry = side_info_cache[frag->id()][node_label];
  if (cache_entry) {
    return cache_entry.get();
  }
  std::cerr << "init node sideinfo " << frag->id();
  auto side_info = std::make_shared<SideInfo>();
  // compute attribute numbers of i/f/s
  auto node_table = frag->vertex_data_table(node_label);
  auto vtable_schema = node_table->schema();
  for (size_t idx = 0; idx < vtable_schema->fields().size() - 1; ++idx) {
    auto field = vtable_schema->fields()[idx];
    switch (field->type()->id()) {
    case arrow::Type::INT64:
      side_info->i_num += 1;
      LOG(INFO) << "matched a int";
      break;
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      side_info->f_num += 1;
      LOG(INFO) << "matched a float";
      break;
    case arrow::Type::STRING:
      side_info->s_num += 1;
      LOG(INFO) << "matched a string";
      break;
    default:
      break;
    }
  }
  side_info->format = kDefault;
  for (size_t idx = 0; idx < vtable_schema->fields().size() - 1; ++idx) {
    // if (field->name() == "label") {
    //   side_info->format |= kLabeled;
    // } else if (field->name() == "weight") {
    //   side_info->format |= kWeighted;
    // } else {
    //   // otherwise we have attributes
    //   side_info->format |= kAttributed;
    // }
    side_info->format |= kAttributed;
  }
  side_info->type = std::to_string(node_label);
  side_info_cache[frag->id()][node_label] = side_info;
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
#endif

GraphStorage *NewVineyardGraphStorage(const std::string &edge_type) {
#if defined(WITH_VINEYARD)
  LOG(INFO) << "create vineyard graph storage: " << WITH_VINEYARD;
  return new VineyardGraphStorage(edge_type);
#else
  return nullptr;
#endif
}

NodeStorage *NewVineyardNodeStorage(const std::string &node_type) {
#if defined(WITH_VINEYARD)
  LOG(INFO) << "create vineyard node storage: " << WITH_VINEYARD;
  return new VineyardNodeStorage(node_type);
#else
  return nullptr;
#endif
}

} // namespace io
} // namespace graphlearn
