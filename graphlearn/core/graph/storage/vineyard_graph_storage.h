#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_GRAPH_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_GRAPH_STORAGE_H_

#include <memory>

#if defined(WITH_VINEYARD)
#include "vineyard/graph/fragment/arrow_fragment.h"
#include "vineyard/graph/fragment/arrow_fragment_group.h"
#endif

#include "graphlearn/core/graph/storage/graph_storage.h"
#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"
#include "graphlearn/include/config.h"

#if defined(WITH_VINEYARD)

namespace graphlearn {
namespace io {

class VineyardGraphStorage : public GraphStorage {
public:
  explicit VineyardGraphStorage(std::string edge_label = "0",
                                std::string const &decorated_edge_view = "",
                                std::string const &use_attrs = "") {
    std::vector<std::string> edge_args;
    boost::algorithm::split(edge_args, decorated_edge_view, boost::is_any_of("|"));
    std::string src_node_type = edge_args[0];
    std::string dst_node_type = edge_args[1];
    std::string edge_view;
    if (edge_args.size() == 3) {
      edge_view = edge_args[2];
    }

    std::cerr << "edge_label = " << edge_label << ": "
              << src_node_type << " -> " << dst_node_type
              << ", from " << GLOBAL_FLAG(VineyardGraphID);

    if (!edge_view.empty()) {
      std::cerr << ", view on '" << edge_view << "'";
    }
    if (!use_attrs.empty()) {
      std::cerr << ", select attributes: '" << use_attrs << "'";
    }
    std::cerr << std::endl;

    VINEYARD_CHECK_OK(client_.Connect(GLOBAL_FLAG(VineyardIPCSocket)));
    auto fg = client_.GetObject<vineyard::ArrowFragmentGroup>(
        GLOBAL_FLAG(VineyardGraphID));
    if (fg == nullptr) {
      throw std::runtime_error("Graph: failed to find the graph");
    }
    // assume 1 worker per server
    for (auto const &kv : fg->Fragments()) {
      if (fg->FragmentLocations().at(kv.first) == client_.instance_id()) {
        frag_ = client_.GetObject<gl_frag_t>(kv.second);
        break;
      }
    }
    if (frag_ == nullptr) {
      throw std::runtime_error("Graph: failed to find a local fragment");
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

    auto elabels = frag_->schema().GetEdgeLabels();
    auto elabel_index = std::find(elabels.begin(), elabels.end(), edge_label);
    if (elabel_index == elabels.end()) {
      throw std::runtime_error(
          "Graph: failed to find edge label in local fragment: " + edge_label);
    } else {
      edge_label_ = elabel_index - elabels.begin();
    }

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
      for (auto const &field: etable->schema()->fields()) {
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
  virtual const IndexList *GetAllInDegrees() const override {
    return get_all_in_degree(frag_, edge_label_);
  }
  virtual const IndexList *GetAllOutDegrees() const override {
    return get_all_out_degree(frag_, edge_label_);
  }
  virtual const IdArray GetAllSrcIds() const override {
    return IdArray(src_lists_.data(), src_lists_.size());
  }
  virtual const IdArray GetAllDstIds() const override {
    return IdArray(dst_lists_.data(), dst_lists_.size());
  }

private:
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
  int index_for_label_ = -1, index_for_weight_ = -1;

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
