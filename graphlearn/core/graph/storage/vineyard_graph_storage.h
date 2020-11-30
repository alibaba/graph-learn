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
                                std::string const &edge_view = "") {
    std::cerr << "edge_label = " << edge_label << ", from "
              << GLOBAL_FLAG(VineyardGraphID);
    if (!edge_view.empty()) {
      std::cerr << ", view on '" << edge_view << "'";
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
    init_src_dst_list(frag_, edge_label_, src_lists_, dst_lists_);
    side_info_ = frag_edge_side_info(frag_, edge_label_);
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
    return frag_->edge_data_table(edge_label_)->num_rows();
  }
  virtual IdType GetSrcId(IdType edge_id) const override {
    return get_edge_src_id(frag_, edge_label_, src_lists_, edge_id);
  }
  virtual IdType GetDstId(IdType edge_id) const override {
    return get_edge_dst_id(frag_, edge_label_, dst_lists_, edge_id);
  }
  virtual float GetEdgeWeight(IdType edge_id) const override {
    if (!side_info_->IsWeighted()) {
      return -1;
    }
    return get_edge_weight(frag_, edge_label_, edge_id);
  }
  virtual int32_t GetEdgeLabel(IdType edge_id) const override {
    if (!side_info_->IsLabeled()) {
      return -1;
    }
    return get_edge_label(frag_, edge_label_, edge_id);
  }
  virtual Attribute GetEdgeAttribute(IdType edge_id) const override {
    if (!side_info_->IsAttributed()) {
      return Attribute();
    }
    return get_edge_attribute(frag_, edge_label_, edge_id);
  }

  virtual Array<IdType> GetNeighbors(IdType src_id) const override {
    return get_all_outgoing_neighbor_nodes(frag_, src_id, edge_label_);
  }
  virtual Array<IdType> GetOutEdges(IdType src_id) const override {
    return get_all_outgoing_neighbor_edges(frag_, src_id, edge_label_);
  }

  virtual IndexType GetInDegree(IdType dst_id) const override {
    return frag_->GetLocalInDegree(vertex_t{static_cast<uint64_t>(dst_id)},
                                   edge_label_);
  }
  virtual IndexType GetOutDegree(IdType src_id) const override {
    return frag_->GetLocalOutDegree(vertex_t{static_cast<uint64_t>(src_id)},
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
  label_id_t edge_label_;
  SideInfo *side_info_ = nullptr;

  // for edge view
  std::string view_label;
  int32_t seed, nsplit, split_begin, split_end;

  std::vector<IdType> src_lists_;
  std::vector<IdType> dst_lists_;
};

} // namespace io
} // namespace graphlearn

#endif

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_GRAPH_STORAGE_H_
