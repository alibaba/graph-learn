#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_

#include "vineyard/graph/fragment/arrow_fragment.h"

#include "graphlearn/core/graph/storage/edge_storage.h"
#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"

namespace graphlearn {
namespace io {

class VineyardEdgeStorage : public EdgeStorage {
 public:
  explicit VineyardEdgeStorage(std::shared_ptr<gl_frag_t> frag) : frag_(frag) {}

  virtual ~VineyardEdgeStorage() = default;

  virtual void SetSideInfo(const SideInfo* info) override {}
  virtual const SideInfo* GetSideInfo() const override {}

  /// Do some re-organization after data fixed.
  virtual void Build() override {}

  /// Get the total edge count after data fixed.
  virtual IdType Size() const override {
    IdType count = 0;
    for (int label_id = 0; label_id < frag_->vertex_label_num(); ++label_id) {
      count += frag_->edge_data_table(label_id)->num_rows();
    }
    return count;
  }

  /// An EDGE is made up of [ src_id, dst_id, weight, label, attributes ].
  /// Insert the value to get an unique id.
  /// If the value is invalid, return -1.
  virtual IdType Add(EdgeValue* value) override {}

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
  virtual const IdList* GetSrcIds() const override {
    int e_label_num = frag_->edge_label_num();
    IdType count = 0;
    for (int label_id = 0; label_id < e_label_num; ++label_id) {
      count += frag_->edge_data_table(label_id)->num_rows();
    }
    auto src_id_list = new IdList();
    src_id_list->reserve(count);
    for (int label_id = 0; label_id < e_label_num; ++label_id) {
      auto arr = frag_->edge_srcs(label_id);
      for (size_t i = 0; i < arr->length(); ++i) {
        src_id_list->emplace_back(arr->Value(i));
      }
    }
    return src_id_list;
  }
  /// Get all the destination node ids, the count of which is the same with
  /// Size(). These ids are not distinct.
  virtual const IdList* GetDstIds() const override {
    int e_label_num = frag_->edge_label_num();
    IdType count = 0;
    for (int label_id = 0; label_id < e_label_num; ++label_id) {
      count += frag_->edge_data_table(label_id)->num_rows();
    }
    auto dst_id_list = new IdList();
    dst_id_list->reserve(count);
    for (int label_id = 0; label_id < e_label_num; ++label_id) {
      auto arr = frag_->edge_dsts(label_id);
      for (size_t i = 0; i < arr->length(); ++i) {
        dst_id_list->emplace_back(arr->Value(i));
      }
    }
    return dst_id_list;
  }
  /// Get all weights if existed, the count of which is the same with Size().
  virtual const std::vector<float>* GetWeights() const override {
    int e_label_num = frag_->edge_label_num();
    IdType count = 0;
    for (int label_id = 0; label_id < e_label_num; ++label_id) {
      count += frag_->edge_data_table(label_id)->num_rows();
    }
    auto weight_list = new std::vector<float>();
    weight_list->reserve(count);
    for (int label_id = 0; label_id < e_label_num; ++label_id) {
      auto table = frag_->edge_data_table(label_id);
      for (size_t i = 0; i < table->num_rows(); ++i) {
        weight_list->emplace_back(static_cast<float>(
            std::dynamic_pointer_cast<
                typename vineyard::ConvertToArrowType<double>::ArrayType>(
                table->column(2)->chunk(0))
                ->Value(i)));
      }
    }
    return weight_list;
  }
  /// Get all labels if existed, the count of which is the same with Size().
  virtual const std::vector<int32_t>* GetLabels() const override {
    int e_label_num = frag_->edge_label_num();
    IdType count = 0;
    for (int label_id = 0; label_id < e_label_num; ++label_id) {
      count += frag_->edge_data_table(label_id)->num_rows();
    }
    auto label_list = new std::vector<int32_t>();
    label_list->reserve(count);
    for (int label_id = 0; label_id < e_label_num; ++label_id) {
      auto table = frag_->edge_data_table(label_id);
      for (size_t i = 0; i < table->num_rows(); ++i) {
        label_list->emplace_back(label_id);
      }
    }
    return label_list;
  }
  /// Get all attributes if existed, the count of which is the same with Size().
  virtual const std::vector<Attribute>* GetAttributes() const override {
    int e_label_num = frag_->edge_label_num();
    IdType count = 0;
    for (int label_id = 0; label_id < e_label_num; ++label_id) {
      count += frag_->edge_data_table(label_id)->num_rows();
    }
    auto attribute_list = new std::vector<Attribute>();
    attribute_list->reserve(count);
    for (int label_id = 0; label_id < e_label_num; ++label_id) {
      auto table = frag_->edge_data_table(label_id);
      for (size_t i = 0; i < table->num_rows(); ++i) {
        attribute_list->emplace_back(arrow_line_to_attribute_value(table, i, 2),
                                     true);
      }
    }
    return attribute_list;
  }

 private:
  std::shared_ptr<gl_frag_t> frag_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_
