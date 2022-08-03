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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_

#if defined(WITH_VINEYARD)
#include "vineyard/graph/fragment/arrow_fragment.vineyard.h"
#include "vineyard/graph/fragment/arrow_fragment_group.h"
#endif

#include "core/graph/storage/edge_storage.h"
#include "core/graph/storage/vineyard_graph_storage.h"
#include "include/config.h"

#if defined(WITH_VINEYARD)

namespace graphlearn {
namespace io {

class VineyardEdgeStorage : public EdgeStorage {
public:
  explicit VineyardEdgeStorage(std::string edge_label = "0",
                               const std::string& decorated_edge_view = "",
                               const std::string& use_attrs = "") {
    graph_ = new VineyardGraphStorage(edge_label, decorated_edge_view, use_attrs);
  }

  virtual ~VineyardEdgeStorage() = default;

  virtual void SetSideInfo(const SideInfo *info) override {}
  virtual const SideInfo *GetSideInfo() const override {
    return graph_->GetSideInfo();
  }

  /// Do some re-organization after data fixed.
  virtual void Build() override {}

  /// Get the total edge count after data fixed.
  virtual IdType Size() const override {
    return graph_->GetEdgeCount();
  }

  /// An EDGE is made up of [ src_id, dst_id, weight, label, attributes ].
  /// Insert the value to get an unique id.
  /// If the value is invalid, return -1.
  virtual IdType Add(EdgeValue *value) override {
    throw std::runtime_error("Not implemented");
  }

  /// Lookup edge infos by edge_id, including
  ///    source node id,
  ///    destination node id,
  ///    edge weight,
  ///    edge label,
  ///    edge attributes
  virtual IdType GetSrcId(IdType edge_id) const override {
    return graph_->GetSrcId(edge_id);
  }
  virtual IdType GetDstId(IdType edge_id) const override {
    return graph_->GetDstId(edge_id);
  }
  virtual float GetWeight(IdType edge_id) const override {
    return graph_->GetEdgeWeight(edge_id);
  }
  virtual int32_t GetLabel(IdType edge_id) const override {
    return graph_->GetEdgeLabel(edge_id);
  }
  virtual Attribute GetAttribute(IdType edge_id) const override {
    return graph_->GetEdgeAttribute(edge_id);
  }

  /// For the needs of traversal and sampling, the data distribution is
  /// helpful. The interface should make it convenient to get the global data.
  ///
  /// Get all the source node ids, the count of which is the same with Size().
  /// These ids are not distinct.
  virtual const IdArray GetSrcIds() const override {
    return graph_->GetAllSrcIds();
  }

  /// Get all the destination node ids, the count of which is the same with
  /// Size(). These ids are not distinct.
  virtual const IdArray GetDstIds() const override {
    return graph_->GetAllDstIds();
  }
  /// Get all weights if existed, the count of which is the same with Size().
  virtual const Array<float> GetWeights() const override {
    if (!graph_->side_info_->IsWeighted()) {
      return Array<float>();
    }
    auto table = graph_->frag_->edge_data_table(graph_->edge_label_);
    if (table->num_rows() == 0 || graph_->index_for_weight_ == -1) {
      return Array<float>();
    }
    auto weight_array = std::dynamic_pointer_cast<
        typename vineyard::ConvertToArrowType<float>::ArrayType>(
        table->column(graph_->index_for_weight_)->chunk(0));
    return Array<float>(weight_array->raw_values(), weight_array->length());
  }

  /// Get all labels if existed, the count of which is the same with Size().
  virtual const Array<int32_t> GetLabels() const override {
    if (!graph_->side_info_->IsLabeled()) {
      return Array<int32_t>();
    }
    auto table = graph_->frag_->edge_data_table(graph_->edge_label_);
    if (table->num_rows() == 0 || graph_->index_for_label_ == -1) {
      return Array<int32_t>();
    }
    auto label_array = std::dynamic_pointer_cast<
        typename vineyard::ConvertToArrowType<int32_t>::ArrayType>(
        table->column(graph_->index_for_label_)->chunk(0));
    return Array<int32_t>(label_array->raw_values(), label_array->length());
  }

  /// Get all attributes if existed, the count of which is the same with Size().
  virtual const std::vector<Attribute> *GetAttributes() const override {
    if (!graph_->side_info_->IsAttributed()) {
      return nullptr;
    }
#ifndef NDEBUG
    std::cerr << "edge: get attributes: edge_label = " << graph_->edge_label_
              << std::endl;
#endif
    auto table = graph_->frag_->edge_data_table(graph_->edge_label_);

    auto value_list = new std::vector<Attribute>();
    value_list->reserve(table->num_rows());
#ifndef NDEBUG
    std::cerr << "edge: get attributes: count = " << table->num_rows() << std::endl;
#endif

    for (size_t offset = 0; offset < table->num_rows(); ++offset) {
      value_list->emplace_back(arrow_line_to_attribute_value(
        offset, graph_->i32_indexes_, graph_->i64_indexes_,
        graph_->f32_indexes_, graph_->f64_indexes_,
        graph_->s_indexes_, graph_->ls_indexes_,
        graph_->edge_table_accessors_), true);
    }
    return value_list;
  }

private:
  VineyardGraphStorage *graph_ = nullptr;
};

} // namespace io
} // namespace graphlearn

#endif

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_
