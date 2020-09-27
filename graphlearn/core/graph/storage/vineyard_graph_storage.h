#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_GRAPH_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_GRAPH_STORAGE_H_

#include <memory>

#include "vineyard/graph/fragment/arrow_fragment.h"

#include "graphlearn/core/graph/storage/graph_storage.h"
#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"

namespace graphlearn {
namespace io {

class VineyardGraphStorage : public GraphStorage {
 public:
  explicit VineyardGraphStorage(std::shared_ptr<gl_frag_t> frag) : frag_(frag) {}
  virtual ~VineyardGraphStorage() = default;

  virtual void Lock() override {}
  virtual void Unlock() override {}

  virtual void SetSideInfo(const SideInfo* info) override {}
  virtual const SideInfo* GetSideInfo() const override { return nullptr; }

  virtual void Add(EdgeValue* value) override {}
  virtual void Build() override {}

  virtual IdType GetEdgeCount() const override {
    IdType count = 0;
    for (int label_id = 0; label_id < frag_->edge_label_num(); ++label_id) {
      count += frag_->edge_data_table(label_id)->num_rows();
    }
    return count;
  }
  virtual IdType GetSrcId(IdType edge_id) const override {
    return get_edge_src_id(frag_, edge_id);
  }
  virtual IdType GetDstId(IdType edge_id) const override {
    return get_edge_dst_id(frag_, edge_id);
  }
  virtual float GetEdgeWeight(IdType edge_id) const override {
    return get_edge_weight(frag_, edge_id);
  }
  virtual int32_t GetEdgeLabel(IdType edge_id) const override {
    return get_edge_label(frag_, edge_id);
  }
  virtual Attribute GetEdgeAttribute(IdType edge_id) const override {
    return get_edge_attribute(frag_, edge_id);
  }

  virtual Array<IdType> GetNeighbors(IdType src_id) const override {
    return get_all_outgoing_neighbor_nodes(frag_, src_id);
  }
  virtual Array<IdType> GetOutEdges(IdType src_id) const override {
    return get_all_outgoing_neighbor_edges(frag_, src_id);
  }

  virtual IndexType GetInDegree(IdType dst_id) const override {
    auto v = vertex_t{dst_id};
    IdType count = 0;
    for (int label_id = 0; label_id < frag_->edge_label_num(); ++label_id) {
      count += frag_->GetLocalInDegree(v, label_id);
    }
    return count;
  }
  virtual IndexType GetOutDegree(IdType src_id) const override {
    auto v = vertex_t{src_id};
    IdType count = 0;
    for (int label_id = 0; label_id < frag_->edge_label_num(); ++label_id) {
      count += frag_->GetLocalOutDegree(v, label_id);
    }
    return count;
  }
  virtual const IndexList* GetAllInDegrees() const override {
    return get_all_in_degree(frag_);
  }
  virtual const IndexList* GetAllOutDegrees() const override {
    return get_all_out_degree(frag_);
  }
  virtual const IdList* GetAllSrcIds() const override {
    return get_all_src_ids(frag_);
  }
  virtual const IdList* GetAllDstIds() const override {
    return get_all_dst_ids(frag_);
  }

 private:
  std::shared_ptr<gl_frag_t> frag_;
};

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_GRAPH_STORAGE_H_
