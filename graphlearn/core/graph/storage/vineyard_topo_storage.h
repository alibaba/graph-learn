#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_TOPO_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_TOPO_STORAGE_H_

#if defined(WITH_VINEYARD)
#include "vineyard/graph/fragment/arrow_fragment.h"
#include "vineyard/graph/fragment/arrow_fragment_group.h"
#endif

#include "graphlearn/core/graph/storage/topo_storage.h"
#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"
#include "graphlearn/include/config.h"

#if defined(WITH_VINEYARD)

namespace graphlearn {
namespace io {

class VineyardTopoStorage : public graphlearn::io::TopoStorage {
public:
  explicit VineyardTopoStorage(std::string edge_label = "0",
                               std::string const &edge_view = "") {
    VINEYARD_CHECK_OK(client_.Connect(GLOBAL_FLAG(VineyardIPCSocket)));
    auto fg = client_.GetObject<vineyard::ArrowFragmentGroup>(
        GLOBAL_FLAG(VineyardGraphID));
    if (fg == nullptr) {
      throw std::runtime_error("Topo: failed to find the graph");
    }
    // assume 1 worker per server
    for (auto const &kv : fg->Fragments()) {
      if (fg->FragmentLocations().at(kv.first) == client_.instance_id()) {
        frag_ = client_.GetObject<gl_frag_t>(kv.second);
        break;
      }
    }
    if (frag_ == nullptr) {
      throw std::runtime_error("Topo: failed to find a local fragment");
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
          "Topo: failed to find edge label in local fragment: " + edge_label);
    } else {
      edge_label_ = elabel_index - elabels.begin();
    }
    init_src_dst_list(frag_, edge_label_, src_lists_, dst_lists_);
  }

  virtual ~VineyardTopoStorage() = default;

  /// Do some re-organization after data fixed.
  virtual void Build(EdgeStorage *edges) override {}

  /// An EDGE is made up of [ src_id, attributes, dst_id ].
  /// Before inserted to the TopoStorage, it should be inserted to
  /// EdgeStorage to get an unique id. And then use the id and value here.
  virtual void Add(IdType edge_id, EdgeValue *value) override {}

  /// Get all the neighbor node ids of a given id.
  virtual Array<IdType> GetNeighbors(IdType src_id) const override {
    return get_all_outgoing_neighbor_nodes(frag_, src_id, edge_label_);
  }
  /// Get all the neighbor edge ids of a given id.
  virtual Array<IdType> GetOutEdges(IdType src_id) const override {
    return get_all_outgoing_neighbor_edges(frag_, src_id, edge_label_);
  }
  /// Get the in-degree value of a given id.
  virtual IndexType GetInDegree(IdType dst_id) const override {
    auto v = vertex_t{static_cast<uint64_t>(dst_id)};
    return frag_->GetLocalInDegree(v, edge_label_);
  }
  /// Get the out-degree value of a given id.
  virtual IndexType GetOutDegree(IdType src_id) const override {
    auto v = vertex_t{static_cast<uint64_t>(src_id)};
    return frag_->GetLocalOutDegree(v, edge_label_);
  }

  /// Get all the distinct ids that appear as the source id of an edge.
  /// For example, 6 edges like
  /// [1 2]
  /// [2 3]
  /// [2 4]
  /// [1 3]
  /// [3 1]
  /// [3 2]
  /// GetAllSrcIds() --> {1, 2, 3}
  virtual const IdArray GetAllSrcIds() const override {
    return IdArray(src_lists_.data(), src_lists_.size());
  }

  /// Get all the distinct ids that appear as the destination id of an edge.
  /// For the above example, GetAllDstIds() --> {2, 3, 4, 1}
  virtual const IdArray GetAllDstIds() const override {
    return IdArray(dst_lists_.data(), dst_lists_.size());
  }

  /// Get the out-degree values of all ids corresponding to GetAllSrcIds().
  /// For the above example, GetAllOutDegrees() --> {2, 2, 2}
  virtual const IndexList *GetAllOutDegrees() const override {
    return get_all_in_degree(frag_, edge_label_);
  }

  /// Get the in-degree values of all ids corresponding to GetAllDstIds().
  /// For the above example, GetAllInDegrees() --> {2, 2, 1, 1}
  virtual const IndexList *GetAllInDegrees() const override {
    return get_all_out_degree(frag_, edge_label_);
  }

private:
  vineyard::Client client_;
  std::shared_ptr<gl_frag_t> frag_;
  label_id_t edge_label_;

  // for edge view
  std::string view_label;
  int32_t seed, nsplit, split_begin, split_end;

  std::vector<IdType> src_lists_;
  std::vector<IdType> dst_lists_;
};

} // namespace io
} // namespace graphlearn

#endif

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_TOPO_STORAGE_H_
