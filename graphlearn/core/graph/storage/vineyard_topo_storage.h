#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_TOPO_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_TOPO_STORAGE_H_

#include "vineyard/graph/fragment/arrow_fragment.h"

#include "graphlearn/core/graph/storage/topo_storage.h"
#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"

namespace graphlearn {
namespace io {

class VineyardTopoStorage : public graphlearn::io::TopoStorage {
 public:
  explicit VineyardTopoStorage(std::shared_ptr<gl_frag_t> frag) : frag_(frag) {}
  virtual ~VineyardTopoStorage() = default;

  /// Do some re-organization after data fixed.
  virtual void Build(EdgeStorage* edges) override {}

  /// An EDGE is made up of [ src_id, attributes, dst_id ].
  /// Before inserted to the TopoStorage, it should be inserted to
  /// EdgeStorage to get an unique id. And then use the id and value here.
  virtual void Add(IdType edge_id, EdgeValue* value) override {}

  /// Get all the neighbor node ids of a given id.
  virtual Array<IdType> GetNeighbors(IdType src_id) const override {
    return get_all_outgoing_neighbor_nodes(frag_, src_id);
  }
  /// Get all the neighbor edge ids of a given id.
  virtual Array<IdType> GetOutEdges(IdType src_id) const override {
    return get_all_outgoing_neighbor_edges(frag_, src_id);
  }
  /// Get the in-degree value of a given id.
  virtual IndexType GetInDegree(IdType dst_id) const override {
    auto v = vertex_t{dst_id};
    IdType count = 0;
    for (int label_id = 0; label_id < frag_->edge_label_num(); ++label_id) {
      count += frag_->GetLocalInDegree(v, label_id);
    }
    return count;
  }
  /// Get the out-degree value of a given id.
  virtual IndexType GetOutDegree(IdType src_id) const override {
    auto v = vertex_t{src_id};
    IdType count = 0;
    for (int label_id = 0; label_id < frag_->edge_label_num(); ++label_id) {
      count += frag_->GetLocalOutDegree(v, label_id);
    }
    return count;
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
  virtual const IdList* GetAllSrcIds() const override {
    return get_all_src_ids(frag_);
  }

  /// Get all the distinct ids that appear as the destination id of an edge.
  /// For the above example, GetAllDstIds() --> {2, 3, 4, 1}
  virtual const IdList* GetAllDstIds() const override {
    return get_all_dst_ids(frag_);
  }

  /// Get the out-degree values of all ids corresponding to GetAllSrcIds().
  /// For the above example, GetAllOutDegrees() --> {2, 2, 2}
  virtual const IndexList* GetAllOutDegrees() const override {
    return get_all_in_degree(frag_);
  }

  /// Get the in-degree values of all ids corresponding to GetAllDstIds().
  /// For the above example, GetAllInDegrees() --> {2, 2, 1, 1}
  virtual const IndexList* GetAllInDegrees() const override {
    return get_all_out_degree(frag_);
  }

 private:
  std::shared_ptr<gl_frag_t> frag_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_TOPO_STORAGE_H_
