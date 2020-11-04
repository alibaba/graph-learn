#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_

#if defined(WITH_VINEYARD)
#include "vineyard/graph/fragment/arrow_fragment.h"
#include "vineyard/graph/fragment/arrow_fragment_group.h"
#endif

#include "graphlearn/core/graph/storage/edge_storage.h"
#include "graphlearn/core/graph/storage/vineyard_graph_storage.h"
#include "graphlearn/include/config.h"

#if defined(WITH_VINEYARD)

namespace graphlearn {
namespace io {

class VineyardEdgeStorage : public EdgeStorage {
public:
  explicit VineyardEdgeStorage(std::string edge_label = "0",
                               std::string const &decorated_edge_view = "",
                               std::string const &use_attrs = "") {
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
  virtual IdType Add(EdgeValue *value) override {}

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
    throw std::runtime_error("Not implemented");
  }

  /// Get all labels if existed, the count of which is the same with Size().
  virtual const Array<int32_t> GetLabels() const override {
    throw std::runtime_error("Not implemented");
  }

  /// Get all attributes if existed, the count of which is the same with Size().
  virtual const std::vector<Attribute> *GetAttributes() const override {
    throw std::runtime_error("Not implemented");
  }

private:
  VineyardGraphStorage *graph_ = nullptr;
};

} // namespace io
} // namespace graphlearn

#endif

#endif // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_EDGE_STORAGE_H_
