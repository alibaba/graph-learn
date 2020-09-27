#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_NODE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_NODE_STORAGE_H_

#include "vineyard/graph/fragment/arrow_fragment.h"

#include "graphlearn/core/graph/storage/node_storage.h"
#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"

namespace graphlearn {
namespace io {

class VineyardNodeStorage : public graphlearn::io::NodeStorage {
 public:
  explicit VineyardNodeStorage(std::shared_ptr<gl_frag_t> const frag)
      : frag_(frag) {}

  virtual ~VineyardNodeStorage() = default;

  virtual void Lock() override {}
  virtual void Unlock() override {}

  virtual void SetSideInfo(const SideInfo* info) override {}
  virtual const SideInfo* GetSideInfo() const override { return nullptr; }

  /// Do some re-organization after data fixed.
  virtual void Build() override {}

  /// Get the total edge count after data fixed.
  virtual IdType Size() const override {
    IdType count = 0;
    for (int label_id = 0; label_id < frag_->vertex_label_num(); ++label_id) {
      count += frag_->GetInnerVerticesNum(label_id);
    }
    return count;
  }

  /// A NODE is made up of [ id, attributes, weight, label ].
  /// Insert a node. If a node with the same id existed, just ignore.
  virtual void Add(NodeValue* value) override {}

  /// Lookup node infos by node_id, including
  ///    node weight,
  ///    node label,
  ///    node attributes
  virtual float GetWeight(IdType node_id) const override {
    // FIXME: hard code property 0 is the weight.
    return static_cast<float>(frag_->GetData<double>(vertex_t{node_id}, 0));
  }

  virtual int32_t GetLabel(IdType node_id) const override {
    return frag_->vertex_label(vertex_t{node_id});
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
  virtual const IdList* GetIds() const override {
    int v_label_num = frag_->vertex_label_num();
    IdType count = 0;
    for (int label_id = 0; label_id < v_label_num; ++label_id) {
      count += frag_->GetInnerVerticesNum(label_id);
    }
    auto id_list = new IdList();
    id_list->reserve(count);
    for (int label_id = 0; label_id < v_label_num; ++label_id) {
      auto id_range = frag_->InnerVertices(label_id);
      for (auto id = id_range.begin(); id < id_range.end(); ++id) {
        id_list->emplace_back(id.GetValue());
      }
    }
    return id_list;
  }

  /// Get all weights if existed, the count of which is the same with Size().
  virtual const std::vector<float>* GetWeights() const override {
    int v_label_num = frag_->vertex_label_num();
    IdType count = 0;
    for (int label_id = 0; label_id < v_label_num; ++label_id) {
      count += frag_->GetInnerVerticesNum(label_id);
    }
    auto weight_list = new std::vector<float>();
    weight_list->reserve(count);
    for (int label_id = 0; label_id < v_label_num; ++label_id) {
      auto id_range = frag_->InnerVertices(label_id);
      for (auto id = id_range.begin(); id < id_range.end(); ++id) {
        weight_list->emplace_back(
            static_cast<float>(frag_->GetData<double>(id, 0)));
      }
    }
    return weight_list;
  }
  /// Get all labels if existed, the count of which is the same with Size().
  virtual const std::vector<int32_t>* GetLabels() const override {
    int v_label_num = frag_->vertex_label_num();
    IdType count = 0;
    for (int label_id = 0; label_id < v_label_num; ++label_id) {
      count += frag_->GetInnerVerticesNum(label_id);
    }
    auto label_list = new std::vector<int32_t>();
    label_list->reserve(count);
    for (int label_id = 0; label_id < v_label_num; ++label_id) {
      for (size_t i = 0; i < frag_->GetInnerVerticesNum(label_id); ++i) {
        label_list->emplace_back(label_id);
      }
    }
    return label_list;
  }
  /// Get all attributes if existed, the count of which is the same with Size().
  virtual const std::vector<Attribute>* GetAttributes() const override {
    int v_label_num = frag_->vertex_label_num();
    IdType count = 0;
    for (int label_id = 0; label_id < v_label_num; ++label_id) {
      count += frag_->GetInnerVerticesNum(label_id);
    }
    auto attr_list = new std::vector<Attribute>();
    attr_list->reserve(count);
    for (int label_id = 0; label_id < v_label_num; ++label_id) {
      auto id_range = frag_->InnerVertices(label_id);
      auto table = frag_->vertex_data_table(label_id);
      for (auto id = id_range.begin(); id < id_range.end(); ++id) {
        auto offset = frag_->vertex_offset(id);
        attr_list->emplace_back(arrow_line_to_attribute_value(table, offset, 0),
                                true);
      }
    }
    return attr_list;
  }

 private:
  std::shared_ptr<gl_frag_t> const frag_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_NODE_STORAGE_H_
