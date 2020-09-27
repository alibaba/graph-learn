#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"

#include <memory>

namespace graphlearn {
namespace io {

AttributeValue *
arrow_line_to_attribute_value(std::shared_ptr<arrow::Table> table,
                              int row_index, int start_index) {
  auto attr = NewDataHeldAttributeValue();
  VINEYARD_ASSERT(row_index < table->num_rows());
  for (int idx = start_index; idx < table->num_columns(); ++idx) {
    auto arr = table->column(idx)->chunk(0);
    switch (arr->type()->id()) {
    case arrow::Type::INT64:
      attr->Add(
          std::dynamic_pointer_cast<
              typename vineyard::ConvertToArrowType<int64_t>::ArrayType>(arr)
              ->Value(row_index));
      break;
    case arrow::Type::FLOAT:
      attr->Add(
          std::dynamic_pointer_cast<
              typename vineyard::ConvertToArrowType<float>::ArrayType>(arr)
              ->Value(row_index));
      break;
    case arrow::Type::DOUBLE:
      attr->Add(static_cast<float>(
          std::dynamic_pointer_cast<
              typename vineyard::ConvertToArrowType<double>::ArrayType>(arr)
              ->Value(row_index)));
      break;
    case arrow::Type::STRING:
      attr->Add(
          std::dynamic_pointer_cast<
              typename vineyard::ConvertToArrowType<std::string>::ArrayType>(
              arr)
              ->GetString(row_index));
      break;
    default:
      LOG(ERROR) << "Unsupported attribute type: " << arr->type()->ToString();
    }
  }
  return attr;
}

const IdList *get_all_src_ids(std::shared_ptr<gl_frag_t> const &frag) {
  int v_label_num = frag->vertex_label_num();
  int e_label_num = frag->edge_label_num();
  auto src_id_list = new IdList();
  for (int label_id = 0; label_id < v_label_num; ++label_id) {
    auto id_range = frag->InnerVertices(label_id);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      for (int e_label = 0; e_label < e_label_num; ++e_label) {
        if (frag->HasChild(id, e_label)) {
          src_id_list->emplace_back(id.GetValue());
        }
      }
    }
  }
  return src_id_list;
}

const IdList *get_all_dst_ids(std::shared_ptr<gl_frag_t> const &frag) {
  int v_label_num = frag->vertex_label_num();
  int e_label_num = frag->edge_label_num();
  auto dst_id_list = new IdList();
  for (int label_id = 0; label_id < v_label_num; ++label_id) {
    auto id_range = frag->InnerVertices(label_id);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      for (int e_label = 0; e_label < e_label_num; ++e_label) {
        if (frag->HasParent(id, e_label)) {
          dst_id_list->emplace_back(id.GetValue());
        }
      }
    }
  }
  return dst_id_list;
}

const IndexList *get_all_in_degree(std::shared_ptr<gl_frag_t> const &frag) {
  int v_label_num = frag->vertex_label_num();
  int e_label_num = frag->edge_label_num();
  auto degree_list = new IndexList();
  for (int label_id = 0; label_id < v_label_num; ++label_id) {
    auto id_range = frag->InnerVertices(label_id);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      for (int e_label = 0; e_label < e_label_num; ++e_label) {
        auto degree = frag->GetLocalInDegree(id, e_label);
        if (degree > 0) {
          degree_list->emplace_back(degree);
        }
      }
    }
  }
  return degree_list;
}

const IndexList *get_all_out_degree(std::shared_ptr<gl_frag_t> const &frag) {
  int v_label_num = frag->vertex_label_num();
  int e_label_num = frag->edge_label_num();
  auto degree_list = new IndexList();
  for (int label_id = 0; label_id < v_label_num; ++label_id) {
    auto id_range = frag->InnerVertices(label_id);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      for (int e_label = 0; e_label < e_label_num; ++e_label) {
        auto degree = frag->GetLocalOutDegree(id, e_label);
        if (degree > 0) {
          degree_list->emplace_back(degree);
        }
      }
    }
  }
  return degree_list;
}

const Array<IdType>
get_all_outgoing_neighbor_nodes(std::shared_ptr<gl_frag_t> const &frag,
                                IdType src_id) {
  using nbr_unit_t = vineyard::property_graph_utils::NbrUnit<gl_frag_t::vid_t,
                                                             gl_frag_t::eid_t>;
  int e_label_num = frag->edge_label_num();
  std::vector<const IdType *> values(e_label_num);
  std::vector<int32_t> sizes(e_label_num);
  for (int label_id = 0; label_id < e_label_num; ++label_id) {
    auto neighbor_list = frag->GetOutgoingAdjList(vertex_t{src_id}, label_id);
    values[label_id] =
        reinterpret_cast<const IdType *>(neighbor_list.raw_begin());
    sizes[label_id] = neighbor_list.Size();
    LOG(INFO) << "label_id = " << label_id << ", sizes = " << sizes[label_id];
  }
  return Array<IdType>(std::make_shared<MultiArray<IdType>>(
      values, sizes, sizeof(nbr_unit_t), __builtin_offsetof(nbr_unit_t, vid)));
}

const Array<IdType>
get_all_outgoing_neighbor_edges(std::shared_ptr<gl_frag_t> const &frag,
                                IdType src_id) {
  using nbr_unit_t = vineyard::property_graph_utils::NbrUnit<gl_frag_t::vid_t,
                                                             gl_frag_t::eid_t>;
  int e_label_num = frag->edge_label_num();
  std::vector<const IdType *> values(e_label_num);
  std::vector<int32_t> sizes(e_label_num);
  for (int label_id = 0; label_id < e_label_num; ++label_id) {
    auto neighbor_list = frag->GetOutgoingAdjList(vertex_t{src_id}, label_id);
    values[label_id] =
        reinterpret_cast<const IdType *>(neighbor_list.raw_begin());
    sizes[label_id] = neighbor_list.Size();
    LOG(INFO) << "label_id = " << label_id << ", sizes = " << sizes[label_id];
  }
  return Array<IdType>(std::make_shared<MultiArray<IdType>>(
      values, sizes, sizeof(nbr_unit_t), __builtin_offsetof(nbr_unit_t, eid)));
}

IdType get_edge_src_id(std::shared_ptr<gl_frag_t> const &frag, IdType edge_id) {
  return frag->edge_src(edge_id);
}

IdType get_edge_dst_id(std::shared_ptr<gl_frag_t> const &frag, IdType edge_id) {
  return frag->edge_dst(edge_id);
}

float get_edge_weight(std::shared_ptr<gl_frag_t> const &frag, IdType edge_id) {
  auto label = frag->edge_label(edge_id);
  auto offset = frag->edge_offset(edge_id);
  auto table = frag->edge_data_table(label);
  return std::dynamic_pointer_cast<
             typename vineyard::ConvertToArrowType<double>::ArrayType>(
             table->column(2)->chunk(0))
      ->Value(offset);
}

int32_t get_edge_label(std::shared_ptr<gl_frag_t> const &frag, IdType edge_id) {
  return frag->edge_label(edge_id);
}

Attribute get_edge_attribute(std::shared_ptr<gl_frag_t> const &frag,
                             IdType edge_id) {
  auto label = frag->edge_label(edge_id);
  auto offset = frag->edge_offset(edge_id);
  auto table = frag->edge_data_table(label);
  return Attribute(arrow_line_to_attribute_value(table, offset, 2), true);
}

} // namespace io
} // namespace graphlearn
