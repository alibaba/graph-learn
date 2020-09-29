#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_

#include "vineyard/graph/fragment/arrow_fragment.h"

#include "graphlearn/core/graph/storage/edge_storage.h"
#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"

namespace graphlearn {
namespace io {

class VineyardEdgeStorage : public EdgeStorage {
public:
  explicit VineyardEdgeStorage(std::shared_ptr<gl_frag_t> frag,
                               label_id_t const edge_label=0)
      : frag_(frag), edge_label_(edge_label) {}

  virtual ~VineyardEdgeStorage() = default;

  virtual void SetSideInfo(const SideInfo *info) override {}
  virtual const SideInfo *GetSideInfo() const override {
    return frag_side_info(frag_, edge_label_);
  }

  /// Do some re-organization after data fixed.
  virtual void Build() override {}

  /// Get the total edge count after data fixed.
  virtual IdType Size() const override {
    return frag_->edge_data_table(edge_label_)->num_rows();
  }

  /// An EDGE is made up of [ src_id, dst_id, weight, label, attributes ].
  /// Insert the value to get an unique id.
  /// If the value is invalid, return -1.
  virtual IdType Add(EdgeValue *value) override {}

  /// Lookup edge infos by edge_id, including
  ///    source node id,
  ///    destination node id,
  ///    edge weight,
  ///    edge label,
  ///    edge attributes
  virtual IdType GetSrcId(IdType edge_id) const override {
    return get_edge_src_id(frag_, edge_id);
  }
  virtual IdType GetDstId(IdType edge_id) const override {
    return get_edge_dst_id(frag_, edge_id);
  }
  virtual float GetWeight(IdType edge_id) const override {
    return get_edge_weight(frag_, edge_id);
  }
  virtual int32_t GetLabel(IdType edge_id) const override {
    return get_edge_label(frag_, edge_id);
  }
  virtual Attribute GetAttribute(IdType edge_id) const override {
    return get_edge_attribute(frag_, edge_id);
  }

  /// For the needs of traversal and sampling, the data distribution is
  /// helpful. The interface should make it convenient to get the global data.
  ///
  /// Get all the source node ids, the count of which is the same with Size().
  /// These ids are not distinct.
  virtual const IdList *GetSrcIds() const override {
    auto edge_srcs = frag_->edge_srcs(edge_label_);
    auto src_id_list = new IdList();
    src_id_list->reserve(edge_srcs->length());
    for (size_t i = 0; i < edge_srcs->length(); ++i) {
      src_id_list->emplace_back(edge_srcs->Value(i));
    }
    return src_id_list;
  }
  /// Get all the destination node ids, the count of which is the same with
  /// Size(). These ids are not distinct.
  virtual const IdList *GetDstIds() const override {
    auto edge_dsts = frag_->edge_dsts(edge_label_);
    auto dst_id_list = new IdList();
    dst_id_list->reserve(edge_dsts->length());
    for (size_t i = 0; i < edge_dsts->length(); ++i) {
      dst_id_list->emplace_back(edge_dsts->Value(i));
    }
    return dst_id_list;
  }
  /// Get all weights if existed, the count of which is the same with Size().
  virtual const std::vector<float> *GetWeights() const override {
    auto table = frag_->edge_data_table(edge_label_);
    auto index = find_index_of_name(table->schema(), "weight");
    if (index == -1) {
      return nullptr;
    }
    auto weight_array = std::dynamic_pointer_cast<
        typename vineyard::ConvertToArrowType<double>::ArrayType>(
            table->column(index)->chunk(0));
    auto weight_list = new std::vector<float>();
    weight_list->reserve(weight_array->length());
    for (size_t i = 0; i < weight_array->length(); ++i) {
      weight_list->emplace_back(static_cast<float>(weight_array->Value(i)));
    }
    return weight_list;
  }
  /// Get all labels if existed, the count of which is the same with Size().
  virtual const std::vector<int32_t> *GetLabels() const override {
    auto table = frag_->edge_data_table(edge_label_);
    auto index = find_index_of_name(table->schema(), "label");
    if (index == -1) {
      return nullptr;
    }
    auto weight_array = std::dynamic_pointer_cast<
        typename vineyard::ConvertToArrowType<double>::ArrayType>(
            table->column(index)->chunk(0));
    auto label_list = new std::vector<int32_t>();
    label_list->reserve(weight_array->length());
    for (size_t i = 0; i < weight_array->length(); ++i) {
      label_list->emplace_back(static_cast<int32_t>(weight_array->Value(i)));
    }
    return label_list;
  }
  /// Get all attributes if existed, the count of which is the same with Size().
  virtual const std::vector<Attribute> *GetAttributes() const override {
    auto table = frag_->edge_data_table(edge_label_);
    auto attribute_list = new std::vector<Attribute>();
    attribute_list->reserve(table->num_rows());
    for (size_t i = 0; i < table->num_rows(); ++i) {
      attribute_list->emplace_back(arrow_line_to_attribute_value(table, i, 2), true);
    }
    return attribute_list;
  }

private:
  std::shared_ptr<gl_frag_t> frag_;
  label_id_t edge_label_;
};

} // namespace io
} // namespace graphlearn

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_
