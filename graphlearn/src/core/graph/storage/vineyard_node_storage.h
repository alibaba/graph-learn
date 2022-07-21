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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_NODE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_NODE_STORAGE_H_

#include <random>
#include <set>

#if defined(WITH_VINEYARD)
#include "vineyard/graph/fragment/arrow_fragment.vineyard.h"
#include "vineyard/graph/fragment/arrow_fragment_group.h"
#endif

#include "core/graph/storage/node_storage.h"
#include "core/graph/storage/vineyard_storage_utils.h"
#include "include/config.h"

#if defined(WITH_VINEYARD)

namespace graphlearn {
namespace io {

class VineyardNodeStorage : public graphlearn::io::NodeStorage {
public:
  explicit VineyardNodeStorage(std::string node_label = "0",
                               std::string const &node_view = "",
                               std::string const &use_attrs = "") {
    std::cerr << "node_label = " << node_label << ", from "
              << GLOBAL_FLAG(VineyardGraphID);
    if (!node_view.empty()) {
      std::cerr << ", view on '" << node_view << "'";
    }
    if (!use_attrs.empty()) {
      std::cerr << ", select attributes: '" << use_attrs << "'";
    }
    std::cerr << std::endl;

    VINEYARD_CHECK_OK(client_.Connect(GLOBAL_FLAG(VineyardIPCSocket)));
    auto fg = client_.GetObject<vineyard::ArrowFragmentGroup>(
        GLOBAL_FLAG(VineyardGraphID));
    if (fg == nullptr) {
      throw std::runtime_error("Node: failed to find the graph");
    }
    // assume 1 worker per server
    for (auto const &kv : fg->Fragments()) {
      if (fg->FragmentLocations().at(kv.first) == client_.instance_id()) {
        frag_ = client_.GetObject<gl_frag_t>(kv.second);
        break;
      }
    }
    if (frag_ == nullptr) {
      throw std::runtime_error("Node: failed to find a local fragment");
    }
    vertex_map_ = frag_->GetVertexMap();

    std::string node_label_name = node_label;

    if (!node_view.empty()) {
      std::vector<std::string> args;
      boost::algorithm::split(args, node_view, boost::is_any_of(":"));
      node_label = args[0];
      seed = stoi(args[1]);
      nsplit = stoi(args[2]);
      split_begin = stoi(args[3]);
      split_end = stoi(args[4]);
    }

    auto const &schema = frag_->schema();
    node_label_ = schema.GetVertexLabelId(node_label);
    if (node_label_ == -1) {
      if (!node_label.empty() && std::all_of(node_label.begin(), node_label.end(), ::isdigit)) {
        node_label_ = std::atoi(node_label.c_str());
      } else {
        throw std::runtime_error(
          "Node: failed to find node label in local fragment: " + node_label);
      }
    }

    auto vtable = frag_->vertex_data_table(node_label_);
    if (use_attrs.empty()) {
      for (auto const &field: vtable->schema()->fields()) {
        attrs_.emplace(field->name());
      }
    } else {
      boost::algorithm::split(attrs_, use_attrs, boost::is_any_of(";"));
    }

    side_info_ = frag_node_side_info(frag_, attrs_, node_label_name, node_label_);
    init_table_accessors(vtable, attrs_, i32_indexes_,
                         i64_indexes_, f32_indexes_, f64_indexes_, s_indexes_,
                         ls_indexes_, vertex_table_accessors_);
    index_for_label_ = find_index_of_name(vtable->schema(), "label");
    index_for_weight_ = find_index_of_name(vtable->schema(), "weight");

    oid_array_ = vertex_map_->GetOidArray(frag_->fid(), node_label_);

    if (node_view.empty()) {
      auto range = frag_->InnerVertices(node_label_);
      #ifndef NDEBUG
          std::cerr << "node: get ids (no view): " << node_label_
                    << ", range begin = " << range.begin_value()
                    << ", range end = " << range.end_value() << std::endl;
      #endif
#if defined(VINEYARD_USE_OID)
      all_ids_ = IdArray(oid_array_->raw_values(), oid_array_->length());
#else
      all_ids_ = IdArray(frag_->GetInnerVertexGid(*range.begin()),
                         frag_->GetInnerVertexGid(*range.end()));
#endif
    } else {
      auto range = frag_->InnerVertices(node_label_);
      #ifndef NDEBUG
          std::cerr << "node: get ids: " << node_label_ << " with view '" << node_view << "'"
                    << ", range begin = " << range.begin_value()
                    << ", range end = " << range.end_value() << std::endl;
      #endif
      std::mt19937 rng(seed);
      std::uniform_int_distribution<> rng_gen(0, nsplit);
      for (auto v: range) {
        int rng_number = rng_gen(rng);
        if (rng_number >= split_begin && rng_number < split_end) {
#if defined(VINEYARD_USE_OID)
          selected_ids_.emplace_back(oid_array_->GetView(frag_->vertex_offset(v)));
#else
          selected_ids_.emplace_back(frag_->GetInnerVertexGid(v));
#endif
        }
      }
      all_ids_ = IdArray(selected_ids_.data(), selected_ids_.size());
    }
  }

  virtual ~VineyardNodeStorage() = default;

  virtual void Lock() override {}
  virtual void Unlock() override {}

  virtual void SetSideInfo(const SideInfo *info) override {}
  virtual const SideInfo *GetSideInfo() const override { return side_info_; }

  /// Do some re-organization after data fixed.
  virtual void Build() override {}

  /// Get the total node count after data fixed.
  virtual IdType Size() const override {
    return all_ids_.Size();
  }

  /// A NODE is made up of [ id, attributes, weight, label ].
  /// Insert a node. If a node with the same id existed, just ignore.
  virtual void Add(NodeValue *value) override {}

  /// Lookup node infos by node_id, including
  ///    node weight,
  ///    node label,
  ///    node attributes
  virtual float GetWeight(IdType node_id) const override {
    if (!side_info_->IsWeighted()) {
      return -1;
    }
#if defined(VINEYARD_USE_OID)
    vineyard_vid_t node_gid;
    if (!vertex_map_->GetGid(frag_->fid(), node_label_, node_id, node_gid)) {
      return -1;
    }
#else
    vineyard_vid_t node_gid = static_cast<vineyard_vid_t>(node_id);
#endif
    auto v = vertex_t{node_gid};
    auto label = frag_->vertex_label(v);
    if (label != node_label_) {
      return -1;
    }
    if (index_for_weight_ == -1) {
      return 0.0;
    }
    auto table = frag_->vertex_data_table(node_label_);
    return static_cast<float>(frag_->GetData<double>(v, index_for_weight_));
  }

  virtual int32_t GetLabel(IdType node_id) const override {
    if (!side_info_->IsLabeled()) {
      return -1;
    }
#if defined(VINEYARD_USE_OID)
    vineyard_vid_t node_gid;
    if (!vertex_map_->GetGid(frag_->fid(), node_label_, node_id, node_gid)) {
      return -1;
    }
#else
    vineyard_vid_t node_gid = static_cast<vineyard_vid_t>(node_id);
#endif
    auto v = vertex_t{node_gid};
    auto label = frag_->vertex_label(v);
    if (label != node_label_) {
      return -1;
    }
    if (index_for_label_ == -1) {
      return -1;
    }
    auto table = frag_->vertex_data_table(node_label_);
    return static_cast<int32_t>(frag_->GetData<int64_t>(v, index_for_label_));
  }

  virtual Attribute GetAttribute(IdType node_id) const override {
    if (!side_info_->IsAttributed()) {
      return Attribute();
    }
#if defined(VINEYARD_USE_OID)
    vineyard_vid_t node_gid;
    if (!vertex_map_->GetGid(frag_->fid(), node_label_, node_id, node_gid)) {
      return Attribute(AttributeValue::Default(side_info_), false);
    }
#else
    vineyard_vid_t node_gid = static_cast<vineyard_vid_t>(node_id);
#endif
    auto v = vertex_t{node_gid};
    if (!frag_->IsInnerVertex(v)) {
      return Attribute(AttributeValue::Default(side_info_), false);
    }
    auto label = frag_->vertex_label(v);
    if (label != node_label_) {
      return Attribute(AttributeValue::Default(side_info_), false);
    }
    auto offset = frag_->vertex_offset(v);
#ifndef NDEBUG
    std::cerr << "node: get attribute: node_id = " << node_id << ", label -> "
              << label << ", offset -> " << offset << std::endl;
#endif

// The thread_local optimization doesn't work for multiple-node graphs,
// will be fixed later.
#if 1
    return Attribute(arrow_line_to_attribute_value(
        offset, i32_indexes_, i64_indexes_, f32_indexes_, f64_indexes_,
        s_indexes_, ls_indexes_, vertex_table_accessors_), true);
#else
    thread_local ArrowRefAttributeValue attribute(
        0, i32_indexes_, i64_indexes_, f32_indexes_, f64_indexes_,
        s_indexes_, ls_indexes_, vertex_table_accessors_);
    attribute.Reuse(offset);
    return Attribute(&attribute, false);
#endif
  }

  /// For the needs of traversal and sampling, the data distribution is
  /// helpful. The interface should make it convenient to get the global data.
  ///
  /// Get all the node ids, the count of which is the same with Size().
  /// These ids are distinct.
  virtual const IdArray GetIds() const override {
    return all_ids_;
  }

  /// Get all weights if existed, the count of which is the same with Size().
  virtual const Array<float> GetWeights() const override {
    if (!side_info_->IsWeighted()) {
      return Array<float>();
    }
    auto table = frag_->vertex_data_table(node_label_);
    if (table->num_rows() == 0 || index_for_weight_ == -1) {
      return Array<float>();
    }
    auto weight_array = std::dynamic_pointer_cast<
        typename vineyard::ConvertToArrowType<float>::ArrayType>(
        table->column(index_for_weight_)->chunk(0));
    return Array<float>(weight_array->raw_values(), weight_array->length());
  }

  /// Get all labels if existed, the count of which is the same with Size().
  virtual const Array<int32_t> GetLabels() const override {
    if (!side_info_->IsLabeled()) {
      return Array<int32_t>();
    }
    auto table = frag_->vertex_data_table(node_label_);
    if (table->num_rows() == 0 || index_for_label_ == -1) {
      return Array<int32_t>();
    }
    auto label_array = std::dynamic_pointer_cast<
        typename vineyard::ConvertToArrowType<int32_t>::ArrayType>(
        table->column(index_for_label_)->chunk(0));
    return Array<int32_t>(label_array->raw_values(), label_array->length());
  }

  /// Get all attributes if existed, the count of which is the same with Size().
  virtual const std::vector<Attribute> *GetAttributes() const override {
    if (!side_info_->IsAttributed()) {
      return nullptr;
    }
#ifndef NDEBUG
    std::cerr << "node: get attributes: node_label = " << node_label_
              << std::endl;
#endif
    size_t count = frag_->GetInnerVerticesNum(node_label_);
#ifndef NDEBUG
    std::cerr << "node: get attributes: count = " << count << std::endl;
#endif

    auto value_list = new std::vector<Attribute>();
    value_list->reserve(count);

    auto id_range = frag_->InnerVertices(node_label_);
    auto vtable = frag_->vertex_data_table(node_label_);
    for (auto id = id_range.begin(); id < id_range.end(); ++id) {
      auto offset = frag_->vertex_offset(*id);
      value_list->emplace_back(arrow_line_to_attribute_value(
        offset, i32_indexes_, i64_indexes_, f32_indexes_, f64_indexes_,
        s_indexes_, ls_indexes_, vertex_table_accessors_), true);
    }
    return value_list;
  }

private:
  vineyard::Client client_;
  std::shared_ptr<gl_frag_t> frag_;
  label_id_t node_label_;
  SideInfo *side_info_ = nullptr;

  // for node view
  std::string view_label_;
  int32_t seed, nsplit, split_begin, split_end;

  IdArray all_ids_;
  std::vector<IdType> selected_ids_;

  // for fast attribute access
  std::vector<int> i32_indexes_, i64_indexes_, f32_indexes_, f64_indexes_,
      s_indexes_, ls_indexes_;
  std::vector<const void *> vertex_table_accessors_;
  int index_for_label_ = -1, index_for_weight_ = -1;

  std::set<std::string> attrs_;
  std::shared_ptr<gl_frag_t::vertex_map_t> vertex_map_;
  std::shared_ptr<vineyard::ConvertToArrowType<vineyard_oid_t>::ArrayType> oid_array_;
};

} // namespace io
} // namespace graphlearn

#endif

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_NODE_STORAGE_H_
