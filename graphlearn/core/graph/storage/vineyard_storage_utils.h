#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_STORAGE_UTILS_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_STORAGE_UTILS_H_

#include <memory>

#include "vineyard/graph/fragment/arrow_fragment.h"

#include "graphlearn/core/graph/storage/edge_storage.h"
#include "graphlearn/core/graph/storage/graph_storage.h"
#include "graphlearn/core/graph/storage/node_storage.h"
#include "graphlearn/core/graph/storage/topo_storage.h"

namespace graphlearn {
namespace io {

using vineyard_oid_t = IdType;
using vineyard_vid_t = IdType;

using gl_frag_t = vineyard::ArrowFragment<vineyard_oid_t, vineyard_vid_t>;
using vertex_t = gl_frag_t::vertex_t;

using graphlearn::io::EdgeStorage;
using graphlearn::io::GraphStorage;
using graphlearn::io::NodeStorage;
using graphlearn::io::TopoStorage;

using graphlearn::io::Attribute;
using graphlearn::io::AttributeValue;
using graphlearn::io::EdgeValue;
using graphlearn::io::IdList;
using graphlearn::io::IndexList;
using graphlearn::io::IndexType;
using graphlearn::io::NewDataHeldAttributeValue;
using graphlearn::io::SideInfo;

AttributeValue* arrow_line_to_attribute_value(
    std::shared_ptr<arrow::Table> table, int row_index, int start_index = 0);

const IdList* get_all_src_ids(std::shared_ptr<gl_frag_t> const& frag);

const IdList* get_all_dst_ids(std::shared_ptr<gl_frag_t> const& frag);

const IndexList* get_all_in_degree(std::shared_ptr<gl_frag_t> const& frag);

const IndexList* get_all_out_degree(std::shared_ptr<gl_frag_t> const& frag);

const Array<IdType> get_all_outgoing_neighbor_nodes(
    std::shared_ptr<gl_frag_t> const& frag, IdType src_id);

const Array<IdType> get_all_outgoing_neighbor_edges(
    std::shared_ptr<gl_frag_t> const& frag, IdType src_id);

IdType get_edge_src_id(std::shared_ptr<gl_frag_t> const& frag,
                           IdType edge_id);

IdType get_edge_dst_id(std::shared_ptr<gl_frag_t> const& frag,
                           IdType edge_id);

float get_edge_weight(std::shared_ptr<gl_frag_t> const& frag, IdType edge_id);

int32_t get_edge_label(std::shared_ptr<gl_frag_t> const& frag, IdType edge_id);

Attribute get_edge_attribute(std::shared_ptr<gl_frag_t> const& frag,
                             IdType edge_id);

}  // namespace io
}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_VINEYARD_STORAGE_UTILS_H_
