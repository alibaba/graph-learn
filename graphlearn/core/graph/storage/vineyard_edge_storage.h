#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_

#if defined(WITH_VINEYARD)
#include "vineyard/graph/fragment/arrow_fragment.h"
#include "vineyard/graph/fragment/arrow_fragment_group.h"
#endif

#include "graphlearn/core/graph/storage/edge_storage.h"
#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"
#include "graphlearn/include/config.h"

#if defined(WITH_VINEYARD)

namespace graphlearn {
namespace io {

class VineyardEdgeStorage : public EdgeStorage {
public:
  explicit VineyardEdgeStorage(std::string const &edge_label = "0") {
    VINEYARD_CHECK_OK(client_.Connect(GLOBAL_FLAG(VineyardIPCSocket)));
    auto fg = client_.GetObject<vineyard::ArrowFragmentGroup>(
        GLOBAL_FLAG(VineyardGraphID));
    if (fg == nullptr) {
      throw std::runtime_error("Edge: failed to find the graph");
    }
    // assume 1 worker per server
    for (auto const &kv : fg->Fragments()) {
      if (fg->FragmentLocations().at(kv.first) == client_.instance_id()) {
        LOG(INFO) << "get object: " << kv.second;
        frag_ = client_.GetObject<gl_frag_t>(kv.second);
        break;
      }
    }
    if (frag_ == nullptr) {
      throw std::runtime_error("Edge: failed to find a local fragment");
    }
    auto elabels = frag_->schema().GetEdgeLabels();
    auto elabel_index = std::find(elabels.begin(), elabels.end(), edge_label);
    if (elabel_index == elabels.end()) {
      throw std::runtime_error(
          "Edge: failed to find edge label in local fragment: " + edge_label);
    } else {
      edge_label_ = elabel_index - elabels.begin();
    }
    initSrcDstList(frag_, edge_label_, src_lists_, dst_lists_);
  }

  explicit VineyardEdgeStorage(label_id_t const edge_label = 0)
      : edge_label_(edge_label) {
    VINEYARD_CHECK_OK(client_.Connect(GLOBAL_FLAG(VineyardIPCSocket)));
    auto fg = client_.GetObject<vineyard::ArrowFragmentGroup>(
        GLOBAL_FLAG(VineyardGraphID));
    if (fg == nullptr) {
      throw std::runtime_error("Edge: failed to find the graph");
    }
    // assume 1 worker per server
    for (auto const &kv : fg->Fragments()) {
      if (fg->FragmentLocations().at(kv.first) == client_.instance_id()) {
        frag_ = client_.GetObject<gl_frag_t>(kv.second);
        break;
      }
    }
    if (frag_ == nullptr) {
      throw std::runtime_error("Edge: failed to find a local fragment");
    }
  }

  virtual ~VineyardEdgeStorage() = default;

  virtual void SetSideInfo(const SideInfo *info) override {}
  virtual const SideInfo *GetSideInfo() const override {
    return frag_edge_side_info(frag_, edge_label_);
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
    return get_edge_src_id(frag_, edge_label_, src_lists_, edge_id);
  }
  virtual IdType GetDstId(IdType edge_id) const override {
    return get_edge_dst_id(frag_, edge_label_, dst_lists_, edge_id);
  }
  virtual float GetWeight(IdType edge_id) const override {
    return get_edge_weight(frag_, edge_label_, edge_id);
  }
  virtual int32_t GetLabel(IdType edge_id) const override {
    return get_edge_label(frag_, edge_label_, edge_id);
  }
  virtual Attribute GetAttribute(IdType edge_id) const override {
    return get_edge_attribute(frag_, edge_label_, edge_id);
  }

  /// For the needs of traversal and sampling, the data distribution is
  /// helpful. The interface should make it convenient to get the global data.
  ///
  /// Get all the source node ids, the count of which is the same with Size().
  /// These ids are not distinct.
  virtual const IdArray GetSrcIds() const override {
    return IdArray(src_lists_.data(), src_lists_.size());
  }

  /// Get all the destination node ids, the count of which is the same with
  /// Size(). These ids are not distinct.
  virtual const IdArray GetDstIds() const override {
    return IdArray(dst_lists_.data(), dst_lists_.size());
  }
  /// Get all weights if existed, the count of which is the same with Size().
  virtual const Array<float> GetWeights() const override {
    auto table = frag_->edge_data_table(edge_label_);
    auto index = find_index_of_name(table->schema(), "weight");
    if (index == -1) {
      return Array<float>();
    }
    auto weight_array = std::dynamic_pointer_cast<
        typename vineyard::ConvertToArrowType<float>::ArrayType>(
        table->column(index)->chunk(0));
    return Array<float>(weight_array->raw_values(), weight_array->length());
  }

  /// Get all labels if existed, the count of which is the same with Size().
  virtual const Array<int32_t> GetLabels() const override {
    auto table = frag_->edge_data_table(edge_label_);
    auto index = find_index_of_name(table->schema(), "label");
    if (index == -1) {
      return Array<int32_t>();
    }
    auto label_array = std::dynamic_pointer_cast<
        typename vineyard::ConvertToArrowType<int32_t>::ArrayType>(
        table->column(index)->chunk(0));
    return Array<int32_t>(label_array->raw_values(), label_array->length());
  }

  /// Get all attributes if existed, the count of which is the same with Size().
  virtual const std::vector<Attribute> *GetAttributes() const override {
    auto table = frag_->edge_data_table(edge_label_);
    std::cerr << "table = " << table->schema()->ToString() << std::endl;
    auto attribute_list = new std::vector<Attribute>();
    attribute_list->reserve(table->num_rows());
    for (size_t i = 0; i < table->num_rows(); ++i) {
      attribute_list->emplace_back(arrow_line_to_attribute_value(table, i, 0),
                                   true);
    }
    return attribute_list;
  }

private:
  vineyard::Client client_;
  std::shared_ptr<gl_frag_t> frag_;
  label_id_t edge_label_;

  std::vector<IdType> src_lists_;
  std::vector<IdType> dst_lists_;
};

} // namespace io
} // namespace graphlearn

#endif

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_
