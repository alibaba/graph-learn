/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

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

#include <cstdint>
#include <typeinfo>

#include "graphlearn/include/client.h"
#include "graphlearn/include/dag_dataset.h"
#include "graphlearn/include/status.h"
#include "graphlearn/python/c/py_bind.h"
#include "graphlearn/python/c/py_wrapper.h"

using namespace graphlearn;

namespace {

void ImportNumpy() {
  import_array1();
}

}  // anonymous namespace

void init_client_module(py::module& m) {
  // Request
  py::class_<OpRequest>(m, "OpRequest").def(py::init<>());
  py::class_<OpResponse>(m, "OpResponse").def(py::init<>());
  DEF_REQ(GetNodesRequest);
  DEF_REQ(GetEdgesRequest);
  DEF_REQ(LookupNodesRequest);
  DEF_REQ(LookupEdgesRequest);
  DEF_REQ(SamplingRequest);
  DEF_REQ(ConditionalSamplingRequest);
  DEF_REQ(AggregatingRequest);
  DEF_REQ(SubGraphRequest);
  DEF_REQ(GetCountRequest);
  DEF_REQ(GetDegreeRequest);

  DEF_RES(GetNodesResponse);
  DEF_RES(GetEdgesResponse);
  DEF_RES(LookupNodesResponse);
  DEF_RES(LookupEdgesResponse);
  DEF_RES(SamplingResponse);
  DEF_RES(AggregatingResponse);
  DEF_RES(SubGraphResponse);
  DEF_RES(GetCountResponse);
  DEF_RES(GetDegreeResponse);

  // Client
  py::class_<Client>(m, "Client")
    .def("stop", &Client::Stop)
    .def("get_nodes",
         CALL_FUNC(GetNodes),
         py::arg("request"),
         py::arg("response"))
    .def("get_edges",
         CALL_FUNC(GetEdges),
         py::arg("request"),
         py::arg("response"))
    .def("lookup_nodes",
         CALL_FUNC(LookupNodes),
         py::arg("request"),
         py::arg("response"))
    .def("lookup_edges",
         CALL_FUNC(LookupEdges),
         py::arg("request"),
         py::arg("response"))
    .def("sample_neighbor",
         CALL_FUNC(Sampling),
         py::arg("request"),
         py::arg("response"))
    .def("agg_nodes",
         CALL_FUNC(Aggregating),
         py::arg("request"),
         py::arg("response"))
    .def("sample_subgraph",
         CALL_FUNC(SubGraph),
         py::arg("request"),
         py::arg("response"))
    .def("get_count",
         CALL_FUNC(GetCount),
         py::arg("request"),
         py::arg("response"))
    .def("get_degree",
         CALL_FUNC(GetDegree),
         py::arg("request"),
         py::arg("response"))
    .def("run_op",
         [](Client& self, OpRequest* req, OpResponse* res) {
           return self.RunOp(req, res);
         },
         py::arg("request"),
         py::arg("response"))
    .def("run_dag",
         [](Client & self, DagDef* dag_def) {
           std::unique_ptr<DagRequest> req(new DagRequest());
           req->ParseFrom(dag_def);
           return self.RunDag(req.get());
         },
         py::arg("dag_def"))
    .def("get_dag_values",
         CALL_FUNC(GetDagValues),
         py::arg("request"),
         py::arg("response"))
    .def("cond_neg_sample",
         [](Client& self, ConditionalSamplingRequest* req, SamplingResponse* res) {
          return self.Sampling(req, res);
        },
        py::arg("request"),
        py::arg("response"))
    .def("get_own_servers",
         [](Client & self) {
           return py::cast(self.GetOwnServers());
         });

  m.def("del_op_request", [](OpRequest* req) { delete req; });
  m.def("del_op_response", [](OpResponse* res) { delete res; });

  // Nodes
  m.def("new_get_nodes_request",
        &new_get_nodes_request,
        py::return_value_policy::reference,
        py::arg("node_type"),
        py::arg("strategy"),
        py::arg("node_from"),
        py::arg("batch_size"),
        py::arg("epoch"));

  m.def("new_get_nodes_response",
        &new_get_nodes_response,
        py::return_value_policy::reference);

  m.def("get_node_ids",
        [](GetNodesResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_node_ids(res));
        },
        py::return_value_policy::reference);

  m.def("new_lookup_nodes_request",
        &new_lookup_nodes_request,
        py::return_value_policy::reference,
        py::arg("node_type"));

  m.def("set_lookup_nodes_request",
      [](LookupNodesRequest* req,
        py::object node_ids) {
        ImportNumpy();
        set_lookup_nodes_request(req, node_ids.ptr());
      });

  m.def("new_lookup_nodes_response",
        &new_lookup_nodes_response,
        py::return_value_policy::reference);

  m.def("get_node_weights",
        [](LookupNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_weights(res));
        },
        py::return_value_policy::reference);

  m.def("get_node_labels",
        [](LookupNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_labels(res));
        },
        py::return_value_policy::reference);

  m.def("get_node_int_attributes",
        [](LookupNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_int_attributes(res));
        },
        py::return_value_policy::reference);

  m.def("get_node_float_attributes",
        [](LookupNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_float_attributes(res));
        },
        py::return_value_policy::reference);

  m.def("get_node_string_attributes",
        [](LookupNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_string_attributes(res));
        },
        py::return_value_policy::reference);

  // Edges
  m.def("new_get_edges_request",
        &new_get_edges_request,
        py::return_value_policy::reference,
        py::arg("edge_type"),
        py::arg("strategy"),
        py::arg("batch_size"),
        py::arg("epoch"));

  m.def("new_get_edges_response",
        &new_get_edges_response,
        py::return_value_policy::reference);

  m.def("get_edge_src_id",
        [](GetEdgesResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_edge_src_id(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_dst_id",
        [](GetEdgesResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_edge_dst_id(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_id",
        [](GetEdgesResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_edge_id(res));
        },
        py::return_value_policy::reference);

  m.def("new_lookup_edges_request",
        &new_lookup_edges_request,
        py::return_value_policy::reference,
        py::arg("edge_type"));

  m.def("set_lookup_edges_request",
        [](LookupEdgesRequest* req, py::object src_ids, py::object edge_ids) {
          ImportNumpy();
          set_lookup_edges_request(req, src_ids.ptr(), edge_ids.ptr());
        });

  m.def("new_lookup_edges_response",
        &new_lookup_edges_response,
        py::return_value_policy::reference);

  m.def("get_edge_weights",
        [](LookupEdgesResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_edge_weights(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_labels",
        [](LookupEdgesResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_edge_labels(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_int_attributes",
        [](LookupEdgesResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_edge_int_attributes(res));
       },
       py::return_value_policy::reference);

  m.def("get_edge_float_attributes",
        [](LookupEdgesResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_edge_float_attributes(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_string_attributes",
        [](LookupEdgesResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_edge_string_attributes(res));
        },
        py::return_value_policy::reference);

  // Sampling
  m.def("new_sampling_request",
        &new_sampling_request,
        py::return_value_policy::reference,
        py::arg("type"),
        py::arg("strategy"),
        py::arg("neighbor_count"),
        py::arg("filter_type"));

  m.def("new_conditional_sampling_request",
        &new_conditional_sampling_request,
        py::return_value_policy::reference,
        py::arg("type"),
        py::arg("strategy"),
        py::arg("neighbor_count"),
        py::arg("dst_node_type"),
        py::arg("batch_share"),
        py::arg("unique"));

  m.def("new_sampling_response",
        &new_sampling_response,
        py::return_value_policy::reference);

  m.def("set_sampling_request",
        [](SamplingRequest* req, py::object src_ids) {
          ImportNumpy();
          set_sampling_request(req, src_ids.ptr());
        });

  m.def("set_conditional_sampling_request_ids",
        [](ConditionalSamplingRequest* req,
           py::object src_ids,
           py::object dst_ids) {
          ImportNumpy();
          set_conditional_sampling_request_ids(req, src_ids.ptr(), dst_ids.ptr());
        });

  m.def("set_conditional_sampling_request_cols",
        [](ConditionalSamplingRequest* req,
           const std::vector<int32_t>& int_cols,
           const std::vector<float>& int_props,
           const std::vector<int32_t>& float_cols,
           const std::vector<float>& float_props,
           const std::vector<int32_t>& str_cols,
           const std::vector<float>& str_props) {
          ImportNumpy();
          set_conditional_sampling_request_cols(req, int_cols, int_props,
                                                float_cols, float_props,
                                                str_cols, str_props);
        });

  m.def("get_sampling_node_ids",
        [](SamplingResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_sampling_node_ids(res));
        },
        py::return_value_policy::reference);

  m.def("get_sampling_edge_ids",
        [](SamplingResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_sampling_edge_ids(res));
        },
        py::return_value_policy::reference);

  m.def("get_sampling_node_degrees",
        [](SamplingResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_sampling_node_degrees(res));
        },
        py::return_value_policy::reference);

  // Aggregating
  m.def("new_aggregating_request",
        &new_aggregating_request,
        py::return_value_policy::reference,
        py::arg("node_type"),
        py::arg("strategy"));

  m.def("set_aggregating_request",
        [](AggregatingRequest* req,
          py::object node_ids,
          py::object segement_ids,
          int32_t num_segments) {
            ImportNumpy();
            set_aggregating_request(
              req, node_ids.ptr(), segement_ids.ptr(), num_segments);
        });

  m.def("new_aggregating_response",
        &new_aggregating_response,
        py::return_value_policy::reference);

  m.def("get_aggregating_nodes",
        [](AggregatingResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_aggregating_nodes(res));
        },
        py::return_value_policy::reference);

  // Subgraph Sampling
  m.def("new_subgraph_request",
        &new_subgraph_request,
        py::return_value_policy::reference,
        py::arg("seed_type"),
        py::arg("nbr_type"),
        py::arg("strategy"),
        py::arg("batch_size"),
        py::arg("epoch"));

  m.def("new_subgraph_response",
        &new_subgraph_response,
        py::return_value_policy::reference);

  m.def("get_node_set",
        [](SubGraphResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_node_set(res));
        },
        py::return_value_policy::reference);

  m.def("get_row_idx",
        [](SubGraphResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_row_idx(res));
        },
        py::return_value_policy::reference);

  m.def("get_col_idx",
        [](SubGraphResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_col_idx(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_set",
        [](SubGraphResponse* res) {
          ImportNumpy();
          CAST_RETURN(get_edge_set(res));
        },
        py::return_value_policy::reference);

  // GetCount
  m.def("new_get_count_request",
        &new_get_count_request,
        py::return_value_policy::reference,
        py::arg("type"),
        py::arg("is_node"));

  m.def("new_get_count_response",
        &new_get_count_response,
        py::return_value_policy::reference);

  m.def("get_count",
        [](GetCountResponse* res) {
          return get_count(res);
        },
        py::return_value_policy::reference);

  // GetDegree
  m.def("new_get_degree_request",
        &new_get_degree_request,
        py::return_value_policy::reference,
        py::arg("edge_type"),
        py::arg("node_from"));

  m.def("set_degree_request",
      [](GetDegreeRequest* req,
        py::object node_ids) {
        ImportNumpy();
        set_degree_request(req, node_ids.ptr());
      });

  m.def("new_get_degree_response",
        &new_get_degree_response,
        py::return_value_policy::reference);

  m.def("get_degree",
        [](GetDegreeResponse* res) {
          CAST_RETURN(get_degree(res));
        },
        py::return_value_policy::reference);

  // dag_dataset
  py::class_<Dataset>(m, "Dataset")
    .def(py::init<Client*, int32_t>())
    .def("close", &Dataset::Close)
    .def("next",
         [](Dataset & self,
            int32_t epoch) {
           return self.Next(epoch);
         },
         py::return_value_policy::reference,
         py::arg("epoch"));

  py::class_<GetDagValuesResponse>(m, "GetDagValuesResponse")
    .def(py::init<>())
    .def("valid",
        [](GetDagValuesResponse & self) {
            return self.Valid();
        });

  m.def("get_dag_value",
        [](GetDagValuesResponse* res, int32_t dag_id,
           const std::string& key) {
          ImportNumpy();
          CAST_RETURN(get_dag_value(res, dag_id, key));
        },
        py::return_value_policy::reference);
  m.def("del_get_dag_value_response", [](GetDagValuesResponse* res) { delete res; });

  // dag
  py::class_<DagDef>(m, "DagDef").def(py::init<>());
  py::class_<DagNodeDef>(m, "DagNodeDef").def(py::init<>());
  py::class_<DagEdgeDef>(m, "DagEdgeDef").def(py::init<>());
  m.def("add_dag_node_int_params",
        [](DagNodeDef* node,
           const std::string& name,
           int32_t value) {
             add_dag_node_int_params(node, name, value);
        });

  m.def("add_dag_node_string_params",
        [](DagNodeDef* node,
           const std::string& name,
           const std::string& value) {
             add_dag_node_string_params(node, name, value);
        });

  m.def("add_dag_node_int_vector_params",
        [](DagNodeDef* node,
           const std::string& name,
           std::vector<int32_t> values) {
             add_dag_node_int_vector_params(node, name, values);
        });

  m.def("add_dag_node_float_vector_params",
        [](DagNodeDef* node,
           const std::string& name,
           std::vector<float> values) {
             add_dag_node_float_vector_params(node, name, values);
        });

  m.def("add_dag_node_in_edge",
        [](DagNodeDef* node, const DagEdgeDef* edge) {
             return add_dag_node_in_edge(node, edge);
        });

  m.def("add_dag_node_out_edge",
        [](DagNodeDef* node,
           const DagEdgeDef* edge) {
            add_dag_node_out_edge(node, edge);
        });

  m.def("add_dag_node",
        [](DagDef* dag,
           const DagNodeDef* node) {
             return add_dag_node(dag, node);
        });

  m.def("new_dag",
        &new_dag,
        py::return_value_policy::reference);

  m.def("debug_string",
        [] (DagDef* dag) {
          return debug_string(dag);
        });

  m.def("set_dag_id",
        [](DagDef* dag,
           int32_t dag_id) {
          set_dag_id(dag, dag_id);
        });

  m.def("new_dag_edge",
        &new_dag_edge,
        py::return_value_policy::reference);

  m.def("set_dag_edge_id",
        [](DagEdgeDef* dag_edge,
           int32_t id) {
            set_dag_edge_id(dag_edge, id);
        });

  m.def("set_dag_edge_src_output",
        [](DagEdgeDef* dag_edge,
           const std::string& src_output) {
            set_dag_edge_src_output(dag_edge, src_output);
        });

  m.def("set_dag_edge_dst_input",
        [](DagEdgeDef* dag_edge,
           const std::string& dst_input) {
            set_dag_edge_dst_input(dag_edge, dst_input);
        });

  m.def("new_dag_node",
        &new_dag_node,
        py::return_value_policy::reference);

  m.def("set_dag_node_id",
        [](DagNodeDef* node, int32_t node_id) {
          set_dag_node_id(node, node_id);
        });

  m.def("set_dag_node_op_name",
        [](DagNodeDef* node, const std::string& op_name) {
          set_dag_node_op_name(node, op_name);
        });
}
