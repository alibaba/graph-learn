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

#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_GRAPH_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_GRAPH_STORAGE_H_

#include <memory>

#if defined(WITH_VINEYARD)
#include "vineyard/graph/fragment/arrow_fragment.vineyard.h"
#include "vineyard/graph/fragment/arrow_fragment_group.h"
#endif

#include "core/graph/storage/graph_storage.h"
#include "core/graph/storage/vineyard_storage_utils.h"
#include "include/config.h"

#if defined(WITH_VINEYARD)

namespace graphlearn {
namespace io {

class VineyardEdgeStorage;

class VineyardGraphStorage : public GraphStorage {
public:
  explicit VineyardGraphStorage(std::string edge_label = "0",
                                const std::string& decorated_edge_view = "",
                                const std::string& use_attrs = "") {
    std::vector<std::string> edge_args;
    if (decorated_edge_view.size() > 0) {
      boost::algorithm::split(edge_args, decorated_edge_view, boost::is_any_of("|"));
    }
    std::string edge_view;
    if (edge_args.size() == 3) {
      edge_view = edge_args[2];
    }

    std::cerr << "edge_label = " << edge_label
              << ", from " << GLOBAL_FLAG(VineyardGraphID);

    if (!edge_view.empty()) {
      std::cerr << ", view on '" << edge_view << "'";
    }
    if (!use_attrs.empty()) {
      std::cerr << ", select attributes: '" << use_attrs << "'";
    }
    std::cerr << std::endl;

    VINEYARD_CHECK_OK(client_.Connect(GLOBAL_FLAG(VineyardIPCSocket)));
    frag_ = get_vineyard_fragment(client_, GLOBAL_FLAG(VineyardGraphID));
    if (frag_ == nullptr) {
      throw std::runtime_error(
        "Graph: failed to find the vineyard fragment: " + std::to_string(GLOBAL_FLAG(VineyardGraphID)));
    }
    vertex_map_ = frag_->GetVertexMap();

    std::string edge_label_name = edge_label;

    if (!edge_view.empty()) {
      std::vector<std::string> args;
      boost::algorithm::split(args, edge_view, boost::is_any_of(":"));
      edge_label = args[0];
      seed = stoi(args[1]);
      nsplit = stoi(args[2]);
      split_begin = stoi(args[3]);
      split_end = stoi(args[4]);
    }

    const auto& schema = frag_->schema();
    edge_label_ = schema.GetEdgeLabelId(edge_label);
    if (edge_label_ == -1) {
      if (!edge_label.empty() && std::all_of(edge_label.begin(), edge_label.end(), ::isdigit)) {
        edge_label_ = std::atoi(edge_label.c_str());
      } else {
        throw std::runtime_error(
          "Graph: failed to find edge label in local fragment: " + edge_label);
      }
    }

    const auto& entry = schema.GetEntry(edge_label_, "EDGE");
    std::string src_node_type = entry.relations[0].first, dst_node_type = entry.relations[0].second;
    if (edge_args.size() == 2) {
      src_node_type = edge_args[0];
      dst_node_type = edge_args[1];
    } else if (edge_args.size() == 1) {
      src_node_type = edge_args[0];
    }
    std::cerr << "edge_label = " << edge_label << ": "
              << src_node_type << " -> " << dst_node_type
              << ", from " << GLOBAL_FLAG(VineyardGraphID) << std::endl;

    auto vlabels = frag_->schema().GetVertexLabels();
    auto src_label_index = std::find(vlabels.begin(), vlabels.end(), src_node_type);
    if (src_label_index == vlabels.end()) {
      throw std::runtime_error(
          "Graph: failed to find src node label in local fragment: " + src_node_type);
    } else {
      src_node_label_ = src_label_index - vlabels.begin();
    }
    auto dst_label_index = std::find(vlabels.begin(), vlabels.end(), dst_node_type);
    if (dst_label_index == vlabels.end()) {
      throw std::runtime_error(
          "Graph: failed to find dst node label in local fragment: " + dst_node_type);
    } else {
      dst_node_label_ = dst_label_index - vlabels.begin();
    }

    auto etable = frag_->edge_data_table(edge_label_);
    if (use_attrs.empty()) {
      for (const auto& field: etable->schema()->fields()) {
        attrs_.emplace(field->name());
      }
    } else {
      boost::algorithm::split(attrs_, use_attrs, boost::is_any_of(";"));
    }

    init_src_dst_list(frag_, edge_label_, src_node_label_, dst_node_label_,
                      src_lists_, dst_lists_, edge_lists_, edge_offsets_);
    side_info_ = frag_edge_side_info(frag_, attrs_, edge_label_name, src_node_type, dst_node_type,
                                     edge_label_);
    init_table_accessors(etable, attrs_, i32_indexes_,
                         i64_indexes_, f32_indexes_, f64_indexes_, s_indexes_,
                         ls_indexes_, edge_table_accessors_);
    index_for_label_ = find_index_of_name(etable->schema(), "label");
    index_for_weight_ = find_index_of_name(etable->schema(), "weight");
  }

  virtual ~VineyardGraphStorage() = default;

  virtual void Lock() override {}
  virtual void Unlock() override {}

  virtual void SetSideInfo(const SideInfo *info) override {}
  virtual const SideInfo *GetSideInfo() const override {
    return side_info_;
  }

  virtual void Add(EdgeValue *value) override {}
  virtual void Build() override {}

  virtual IdType GetEdgeCount() const override {
    return edge_lists_.size();
  }
  virtual IdType GetSrcId(IdType edge_id) const override {
    return get_edge_src_id(frag_, edge_label_, src_lists_, edge_id);
  }
  virtual IdType GetDstId(IdType edge_id) const override {
    return get_edge_dst_id(frag_, edge_label_, dst_lists_, edge_id);
  }
  virtual IdType GetEdgeId(IdType edge_index) const override {
    // TODO: @LiSu, check what edge index is in vineyard store
    return edge_index;
  }
  virtual float GetEdgeWeight(IdType edge_id) const override {
    if (!side_info_->IsWeighted() || edge_id >= edge_lists_.size()) {
      return -1;
    }
    return get_edge_weight(frag_, edge_label_, edge_lists_[edge_id]);
  }
  virtual int32_t GetEdgeLabel(IdType edge_id) const override {
    if (!side_info_->IsLabeled() || edge_id >= edge_lists_.size()) {
      return -1;
    }
    return get_edge_label(frag_, edge_label_, edge_lists_[edge_id]);
  }
  virtual int64_t GetEdgeTimestamp(IdType edge_id) const override {
    if (!side_info_->IsTimestamped() || edge_id >= edge_lists_.size()) {
      return -1;
    }
    return get_edge_timestamp(frag_, edge_label_, edge_lists_[edge_id]);
  }
  virtual Attribute GetEdgeAttribute(IdType edge_id) const override {
    if (!side_info_->IsAttributed()) {
      return Attribute();
    }
    if (edge_id >= edge_lists_.size()) {
      return Attribute(AttributeValue::Default(side_info_), false);
    }
    return Attribute(arrow_line_to_attribute_value(
        edge_lists_[edge_id], i32_indexes_, i64_indexes_, f32_indexes_, f64_indexes_,
        s_indexes_, ls_indexes_, edge_table_accessors_), true);
  }

  virtual Array<IdType> GetNeighbors(IdType src_id) const override {
#if defined(VINEYARD_USE_OID)
    vineyard_vid_t src_gid;
    if (!vertex_map_->GetGid(frag_->fid(), src_node_label_, src_id, src_gid)) {
      return IdArray();
    }
#else
    vineyard_vid_t src_gid = static_cast<vineyard_vid_t>(src_id);
#endif
    return get_all_outgoing_neighbor_nodes(frag_, dst_lists_, src_gid, edge_label_,
                                           edge_offsets_);
  }

  virtual Array<IdType> GetOutEdges(IdType src_id) const override {
#if defined(VINEYARD_USE_OID)
    vineyard_vid_t src_gid;
    if (!vertex_map_->GetGid(frag_->fid(), src_node_label_, src_id, src_gid)) {
      return IdArray();
    }
#else
    vineyard_vid_t src_gid = static_cast<vineyard_vid_t>(src_id);
#endif
    return get_all_outgoing_neighbor_edges(frag_, edge_lists_, src_gid, edge_label_,
                                           edge_offsets_);
  }

  virtual IndexType GetInDegree(IdType dst_id) const override {
#if defined(VINEYARD_USE_OID)
    vineyard_vid_t dst_gid;
    if (!vertex_map_->GetGid(frag_->fid(), src_node_label_, dst_id, dst_gid)) {
      return -1;
    }
#else
    vineyard_vid_t dst_gid = static_cast<vineyard_vid_t>(dst_id);
#endif
    return frag_->GetLocalInDegree(vertex_t{static_cast<uint64_t>(dst_gid)},
                                   edge_label_);
  }

  virtual IndexType GetOutDegree(IdType src_id) const override {
#if defined(VINEYARD_USE_OID)
    vineyard_vid_t src_gid;
    if (!vertex_map_->GetGid(frag_->fid(), src_node_label_, src_id, src_gid)) {
      return -1;
    }
#else
    vineyard_vid_t src_gid = static_cast<vineyard_vid_t>(src_id);
#endif
    return frag_->GetLocalOutDegree(vertex_t{static_cast<uint64_t>(src_gid)},
                                    edge_label_);
  }
  virtual const IndexArray GetAllInDegrees() const override {
    return get_all_in_degree(frag_, edge_label_);
  }
  virtual const IndexArray GetAllOutDegrees() const override {
    return get_all_out_degree(frag_, edge_label_);
  }
  virtual const IdArray GetAllSrcIds() const override {
    return IdArray(src_lists_.data(), src_lists_.size());
  }
  virtual const IdArray GetAllDstIds() const override {
    return IdArray(dst_lists_.data(), dst_lists_.size());
  }

private:
  friend class VineyardEdgeStorage;

  vineyard::Client client_;
  std::shared_ptr<gl_frag_t> frag_;
  label_id_t edge_label_, src_node_label_, dst_node_label_;
  SideInfo *side_info_ = nullptr;

  // for edge view
  int32_t seed, nsplit, split_begin, split_end;

  std::set<std::string> attrs_;

  // for fast attribute access
  std::vector<int> i32_indexes_, i64_indexes_, f32_indexes_, f64_indexes_,
      s_indexes_, ls_indexes_;
  std::vector<const void *> edge_table_accessors_;
  int index_for_label_ = -1, index_for_weight_ = -1, index_for_timestamp_ = -1;

  std::vector<IdType> src_lists_;
  std::vector<IdType> dst_lists_;
  std::vector<IdType> edge_lists_;
  std::vector<std::pair<IdType, IdType>> edge_offsets_;

  std::shared_ptr<gl_frag_t::vertex_map_t> vertex_map_;
};

} // namespace io
} // namespace graphlearn

#endif

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_GRAPH_STORAGE_H_
