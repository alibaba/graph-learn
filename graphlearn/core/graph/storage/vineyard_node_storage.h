#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_NODE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_NODE_STORAGE_H_

#include "vineyard/graph/fragment/arrow_fragment.h"

#include "graphlearn/core/graph/storage/node_storage.h"
#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"

namespace graphlearn {
namespace io {

class VineyardNodeStorage : public graphlearn::io::NodeStorage {
public:
  explicit VineyardNodeStorage(label_id_t const edge_label=0)
      : edge_label_(edge_label) {
    VINEYARD_CHECK_OK(client_.Connect(GLOBAL_FLAG(VineyardIPCSocket)));
    frag_ = client_.GetObject<gl_frag_t>(GLOBAL_FLAG(VineyardGraphID));
  }

  virtual ~VineyardNodeStorage() = default;

  virtual void Lock() override {}
  virtual void Unlock() override {}

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

  /// A NODE is made up of [ id, attributes, weight, label ].
  /// Insert a node. If a node with the same id existed, just ignore.
  virtual void Add(NodeValue *value) override {}

  /// Lookup node infos by node_id, including
  ///    node weight,
  ///    node label,
  ///    node attributes
  virtual float GetWeight(IdType node_id) const override {
    auto v = vertex_t{node_id};
    auto table = frag_->vertex_data_table(frag_->vertex_label(v));
    int index = find_index_of_name(table->schema(), "weight");
    return static_cast<float>(frag_->GetData<double>(vertex_t{node_id}, index));
  }

  virtual int32_t GetLabel(IdType node_id) const override {
    auto v = vertex_t{node_id};
    auto table = frag_->vertex_data_table(frag_->vertex_label(v));
    int index = find_index_of_name(table->schema(), "label");
    return static_cast<float>(frag_->GetData<int64_t>(vertex_t{node_id}, index));
  }

  virtual Attribute GetAttribute(IdType node_id) const override {
    auto v = vertex_t{node_id};
    auto label = frag_->vertex_label(v);
    auto offset = frag_->vertex_offset(v);
    auto table = frag_->vertex_data_table(label);
    return Attribute(arrow_line_to_attribute_value(table, offset, 0), true);
  }

  /// For the needs of traversal and sampling, the data distribution is
  /// helpful. The interface should make it convenient to get the global data.
  ///
  /// Get all the node ids, the count of which is the same with Size().
  /// These ids are distinct.
  virtual const IdList *GetIds() const override {
    auto src_v_label = resolve_src_label();
    auto dst_v_label = resolve_dst_label();

    size_t count = frag_->GetInnerVerticesNum(src_v_label);
    if (src_v_label != dst_v_label) {
      count += frag_->GetInnerVerticesNum(dst_v_label);
    }
    auto id_list = new IdList();
    id_list->reserve(count);

    auto src_id_range = frag_->InnerVertices(src_v_label);
    for (auto id = src_id_range.begin(); id < src_id_range.end(); ++id) {
      id_list->emplace_back(id.GetValue());
    }
    if (src_v_label != dst_v_label) {
      auto dst_id_range = frag_->InnerVertices(dst_v_label);
      for (auto id = dst_id_range.begin(); id < dst_id_range.end(); ++id) {
        id_list->emplace_back(id.GetValue());
      }
    }
    return id_list;
  }

  /// Get all weights if existed, the count of which is the same with Size().
  virtual const std::vector<float> *GetWeights() const override {
    return GetAttribute<double, float>("weight");
  }

  /// Get all labels if existed, the count of which is the same with Size().
  virtual const std::vector<int32_t> *GetLabels() const override {
    return GetAttribute<int64_t, int32_t>("label");
  }
  /// Get all attributes if existed, the count of which is the same with Size().
  virtual const std::vector<Attribute> *GetAttributes() const override {
    auto src_v_label = resolve_src_label();
    auto dst_v_label = resolve_dst_label();

    size_t count = 0;
    count += frag_->GetInnerVerticesNum(src_v_label);
    if (src_v_label != dst_v_label) {
      count += frag_->GetInnerVerticesNum(dst_v_label);
    }
    auto id_list = new IdList();
    id_list->reserve(count);

    auto value_list = new std::vector<Attribute>();
    value_list->reserve(count);

    auto id_range = frag_->InnerVertices(src_v_label);
    auto src_v_table = frag_->vertex_data_table(src_v_label);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      auto offset = frag_->vertex_offset(id);
      value_list->emplace_back(
          arrow_line_to_attribute_value(src_v_table, offset, 0), true);
    }
    if (src_v_label != dst_v_label) {
      auto id_range = frag_->InnerVertices(dst_v_label);
      auto dst_v_table = frag_->vertex_data_table(dst_v_label);
      for (auto id = id_range.begin(); id < id_range.end(); ++id) {
        auto offset = frag_->vertex_offset(id);
        value_list->emplace_back(
            arrow_line_to_attribute_value(dst_v_table, offset, 0), true);
      }
    }
    return value_list;
  }

private:
  label_id_t resolve_src_label() const {
    return frag_->vertex_label(vertex_t{frag_->edge_srcs(edge_label_)->Value(0)});
  }
  label_id_t resolve_dst_label() const {
    return frag_->vertex_label(vertex_t{frag_->edge_dsts(edge_label_)->Value(0)});
  }

  template <typename T, typename RT=T>
  const std::vector<RT> *GetAttribute(std::string const &name) const {
    auto src_v_label = resolve_src_label();
    auto dst_v_label = resolve_dst_label();

    int src_index = find_index_of_name(
        frag_->vertex_data_table(src_v_label)->schema(), name);
    int dst_index = find_index_of_name(
        frag_->vertex_data_table(dst_v_label)->schema(), name);

    size_t count = 0;
    if (src_index != -1) {
      count += frag_->GetInnerVerticesNum(src_v_label);
    }
    if (dst_index != -1 && src_v_label != dst_v_label) {
      count += frag_->GetInnerVerticesNum(dst_v_label);
    }
    if (count == 0) {
      return nullptr;
    }
    auto id_list = new IdList();
    id_list->reserve(count);

    auto value_list = new std::vector<RT>();
    value_list->reserve(count);

    if (src_index != -1) {
      auto id_range = frag_->InnerVertices(src_v_label);
      for (auto id = id_range.begin(); id < id_range.end(); ++id) {
        value_list->emplace_back(frag_->GetData<T>(id, src_index));
      }
    }
    if (dst_index != -1 && src_v_label != dst_v_label) {
      auto id_range = frag_->InnerVertices(dst_v_label);
      for (auto id = id_range.begin(); id < id_range.end(); ++id) {
        value_list->emplace_back(frag_->GetData<T>(id, dst_index));
      }
    }
    return value_list;
  }

  vineyard::Client client_;
  std::shared_ptr<gl_frag_t> const frag_;
  label_id_t edge_label_;
};

} // namespace io
} // namespace graphlearn

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_NODE_STORAGE_H_
