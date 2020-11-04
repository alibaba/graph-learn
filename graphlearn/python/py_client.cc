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
#include "graphlearn/include/status.h"
#include "graphlearn/python/py_bind.h"
#include "graphlearn/python/py_wrapper.h"

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
  DEF_REQ(AggregatingRequest);

  DEF_RES(GetNodesResponse);
  DEF_RES(GetEdgesResponse);
  DEF_RES(LookupNodesResponse);
  DEF_RES(LookupEdgesResponse);
  DEF_RES(SamplingResponse);
  DEF_RES(AggregatingResponse);

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
    .def("run_op",
         [](Client & self, OpRequest* req, OpResponse* res) {
           return self.RunOp(req, res);
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
        py::arg("neighbor_count"));

  m.def("new_sampling_response",
        &new_sampling_response,
        py::return_value_policy::reference);

  m.def("set_sampling_request",
        [](SamplingRequest* req, py::object src_ids) {
          ImportNumpy();
          set_sampling_request(req, src_ids.ptr());
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
}
