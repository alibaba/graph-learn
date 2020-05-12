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
#include <iostream>
#include <typeinfo>

#include "graphlearn/include/client.h"
#include "graphlearn/include/config.h"
#include "graphlearn/include/data_source.h"
#include "graphlearn/include/server.h"
#include "graphlearn/include/topology.h"
#include "graphlearn/include/status.h"
#include "graphlearn/python/py_wrapper.h"

using graphlearn::error::Code;

using graphlearn::io::DataFormat;
using graphlearn::io::NodeSource;
using graphlearn::io::EdgeSource;
using graphlearn::io::Direction;

using graphlearn::DataType;
using graphlearn::NodeFrom;
using graphlearn::PartitionMode;
using graphlearn::PaddingMode;

using graphlearn::Server;
using graphlearn::Status;
using graphlearn::NewServer;

using graphlearn::Client;
using graphlearn::NewInMemoryClient;
using graphlearn::NewRpcClient;
using graphlearn::SamplingRequest;
using graphlearn::SamplingResponse;
using graphlearn::GetEdgesRequest;
using graphlearn::GetEdgesResponse;
using graphlearn::GetNodesRequest;
using graphlearn::GetNodesResponse;
using graphlearn::LookupNodesRequest;
using graphlearn::LookupNodesResponse;
using graphlearn::LookupEdgesRequest;
using graphlearn::LookupEdgesResponse;
using graphlearn::AggregatingRequest;
using graphlearn::AggregatingResponse;
using graphlearn::GetTopologyRequest;
using graphlearn::GetTopologyResponse;


PYBIND11_MODULE(pywrap_graphlearn, m) {
  m.doc() = "Python interface for graph-learn.";

  // global flag settings.
  m.def("set_default_neighbor_id", &graphlearn::SetGlobalFlagDefaultNeighborId);
  m.def("set_padding_mode", &graphlearn::SetGlobalFlagPaddingMode);
  m.def("set_default_int_attr", &graphlearn::SetGlobalFlagDefaultIntAttribute);
  m.def("set_default_float_attr",
        &graphlearn::SetGlobalFlagDefaultFloatAttribute);
  m.def("set_default_string_attr",
        &graphlearn::SetGlobalFlagDefaultStringAttribute);
  m.def("set_inmemory_queuesize", &graphlearn::SetGlobalFlagInMemoryQueueSize);
  m.def("set_inner_threadnum", &graphlearn::SetGlobalFlagInterThreadNum);
  m.def("set_inter_threadnum", &graphlearn::SetGlobalFlagInterThreadNum);
  m.def("set_intra_threadnum", &graphlearn::SetGlobalFlagIntraThreadNum);
  m.def("set_datainit_batchsize", &graphlearn::SetGlobalFlagDataInitBatchSize);
  m.def("set_shuffle_buffer_size", &graphlearn::SetGlobalFlagShuffleBufferSize);
  m.def("set_rpc_message_max_size", &graphlearn::SetGlobalFlagRpcMessageMaxSize);
  m.def("set_deploy_mode", &graphlearn::SetGlobalFlagDeployMode);
  m.def("set_client_id", &graphlearn::SetGlobalFlagClientId);
  m.def("set_client_count", &graphlearn::SetGlobalFlagClientCount);
  m.def("set_server_count", &graphlearn::SetGlobalFlagServerCount);
  m.def("set_tracker", &graphlearn::SetGlobalFlagTracker);

  // protobuf
  // erro code
  py::enum_<Code>(m, "ErrorCode")
    .value("OK", Code::OK)
    .value("CANCELLED", Code::CANCELLED)
    .value("UNKNOWN", Code::UNKNOWN)
    .value("INVALID_ARGUMENT", Code::INVALID_ARGUMENT)
    .value("DEADLINE_EXCEEDED", Code::DEADLINE_EXCEEDED)
    .value("NOT_FOUND", Code::NOT_FOUND)
    .value("ALREADY_EXISTS", Code::ALREADY_EXISTS)
    .value("PERMISSION_DENIED", Code::PERMISSION_DENIED)
    .value("UNAUTHENTICATED", Code::UNAUTHENTICATED)
    .value("RESOURCE_EXHAUSTED", Code::RESOURCE_EXHAUSTED)
    .value("FAILED_PRECONDITION", Code::FAILED_PRECONDITION)
    .value("ABORTED", Code::ABORTED)
    .value("OUT_OF_RANGE", Code::OUT_OF_RANGE)
    .value("UNIMPLEMENTED", Code::UNIMPLEMENTED)
    .value("INTERNAL", Code::INTERNAL)
    .value("UNAVAILABLE", Code::UNAVAILABLE)
    .value("DATA_LOSS", Code::DATA_LOSS)
    .value("REQUEST_STOP", Code::REQUEST_STOP);
  
  py::enum_<NodeFrom>(m, "NodeFrom")		
    .value("EDGE_SRC", NodeFrom::kEdgeSrc)		
    .value("EDGE_DST", NodeFrom::kEdgeDst)		
    .value("NODE", NodeFrom::kNode);

  // data source
  py::enum_<DataFormat>(m, "DataFormat")
    .value("DEFAULT", DataFormat::kDefault)
    .value("WEIGHTED", DataFormat::kWeighted)
    .value("LABELED", DataFormat::kLabeled)
    .value("ATTRIBUTED", DataFormat::kAttributed);

  py::enum_<DataType>(m, "DataType")
    .value("INT32", DataType::kInt32)
    .value("INT64", DataType::kInt64)
    .value("FLOAT", DataType::kFloat)
    .value("DOUBLE", DataType::kDouble)
    .value("STRING", DataType::kString);

  py::enum_<PartitionMode>(m, "PartitionMode")
    .value("NO_PARTITION", PartitionMode::kNoPartition)
    .value("BY_SOURCE_ID", PartitionMode::kByHash);

  py::enum_<PaddingMode>(m, "PaddingMode")
    .value("REPLICATE", PaddingMode::kReplicate)
    .value("CIRCULAR", PaddingMode::kCircular);

  py::enum_<Direction>(m, "Direction")
    .value("ORIGIN", Direction::kOrigin)
    .value("REVERSED", Direction::kReversed);

  py::class_<NodeSource>(m, "NodeSource")
    .def(py::init<>())
    .def_readwrite("path", &NodeSource::path)
    .def_readwrite("format", &NodeSource::format)
    .def_readwrite("id_type", &NodeSource::id_type)
    .def_readwrite("attr_types", &NodeSource::types)
    .def_readwrite("delimiter", &NodeSource::delimiter)
    .def_readwrite("hash_buckets", &NodeSource::hash_buckets)
    .def_readwrite("ignore_invalid", &NodeSource::ignore_invalid)
    .def("append_attr_type", &NodeSource::AppendAttrType)
    .def("append_hash_bucket", &NodeSource::AppendHashBucket);

  py::class_<EdgeSource>(m, "EdgeSource")
    .def(py::init<>())
    .def_readwrite("path", &EdgeSource::path)
    .def_readwrite("format", &EdgeSource::format)
    .def_readwrite("edge_type", &EdgeSource::edge_type)
    .def_readwrite("src_id_type", &EdgeSource::src_id_type)
    .def_readwrite("dst_id_type", &EdgeSource::dst_id_type)
    .def_readwrite("attr_types", &EdgeSource::types)
    .def_readwrite("delimiter", &EdgeSource::delimiter)
    .def_readwrite("hash_buckets", &EdgeSource::hash_buckets)
    .def_readwrite("ignore_invalid", &EdgeSource::ignore_invalid)
    .def_readwrite("direction", &EdgeSource::direction)
    .def("append_attr_type", &EdgeSource::AppendAttrType)
    .def("append_hash_bucket", &EdgeSource::AppendHashBucket);

  // Status
  py::class_<Status>(m, "Status")
    .def("ok", &Status::ok)
    .def("code", &Status::code)
    .def("message", &Status::msg)
    .def("to_string", &Status::ToString);

  // New client and Server object.
  m.def("server",
        &NewServer,
        py::return_value_policy::take_ownership,
        py::arg("server_id"),
        py::arg("server_count"),
        py::arg("tracker"));

  // Server methods.
  py::class_<Server>(m, "Server")
    .def("start", &Server::Start)
    .def("init", &Server::Init)
    .def("stop", &Server::Stop);

  //////////////////////////// NEW RPC ///////////////////////////////////
  m.def("in_memory_client",
        &NewInMemoryClient,
        py::return_value_policy::take_ownership);
  m.def("rpc_client",
        &NewRpcClient,
        py::return_value_policy::take_ownership,
        py::arg("server_id") = -1,
        py::arg("server_own") = false);

  py::class_<SamplingRequest>(m, "SamplingRequest")
    .def(py::init<>());
  py::class_<SamplingResponse>(m, "SamplingResponse")
    .def(py::init<>());
  py::class_<GetEdgesRequest>(m, "GetEdgesRequest")
    .def(py::init<>());
  py::class_<GetEdgesResponse>(m, "GetEdgesResponse")
    .def(py::init<>());
  py::class_<GetNodesRequest>(m, "GetNodesRequest")
    .def(py::init<>());
  py::class_<GetNodesResponse>(m, "GetNodesResponse")
    .def(py::init<>());
  py::class_<LookupEdgesRequest>(m, "LookupEdgesRequest")
    .def(py::init<>());
  py::class_<LookupEdgesResponse>(m, "LookupEdgesResponse")
    .def(py::init<>());
  py::class_<LookupNodesRequest>(m, "LookupNodesRequest")
    .def(py::init<>());
  py::class_<LookupNodesResponse>(m, "LookupNodesResponse")
    .def(py::init<>());
  py::class_<AggregatingRequest>(m, "AggregatingRequest")
    .def(py::init<>());
  py::class_<AggregatingResponse>(m, "AggregatingResponse")
    .def(py::init<>());

  // Client methods.
  py::class_<Client>(m, "Client")
    .def("stop", &Client::Stop)
    .def("sample_neighbor",
        [](Client & self, SamplingRequest* req,
          SamplingResponse* res) {
          Status s = self.Sampling(req, res);
          return s;
        },
        py::arg("request"),
        py::arg("response"))
    .def("get_edges",
        [](Client & self, GetEdgesRequest* req,
          GetEdgesResponse* res) {
          Status s = self.GetEdges(req, res);
          return s;
        },
        py::arg("request"),
        py::arg("response"))
    .def("lookup_edges",
        [](Client & self, LookupEdgesRequest* req,
          LookupEdgesResponse* res) {
          Status s = self.LookupEdges(req, res);
          return s;
        },
        py::arg("request"),
        py::arg("response"))
    .def("get_nodes",
        [](Client & self, GetNodesRequest* req,
          GetNodesResponse* res) {
          Status s = self.GetNodes(req, res);
          return s;
        },
        py::arg("request"),
        py::arg("response"))
    .def("lookup_nodes",
        [](Client & self, LookupNodesRequest* req,
          LookupNodesResponse* res) {
          Status s = self.LookupNodes(req, res);
          return s;
        },
        py::arg("request"),
        py::arg("response"))
    .def("agg_nodes",
        [](Client & self, AggregatingRequest* req,
          AggregatingResponse* res) {
          Status s = self.Aggregating(req, res);
          return s;
        },
        py::arg("request"),
        py::arg("response"));

  // Neighbors
  m.def("new_nbr_req",
        &new_nbr_req,
        py::return_value_policy::reference,
        py::arg("type"),
        py::arg("strategy"),
        py::arg("neighbor_count"));

  m.def("del_nbr_req",
        [](SamplingRequest* req) {
          del_nbr_req(req);
        });

  m.def("new_nbr_res",
        &new_nbr_res,
        py::return_value_policy::reference);

  m.def("del_nbr_res",
        [](SamplingResponse* res) {
          del_nbr_res(res);
        });

  m.def("set_nbr_req",
        [](SamplingRequest* req,
           py::object src_ids) {
          ImportNumpy();
          set_nbr_req(req, src_ids.ptr());
        });

  m.def("get_nbr_res_nbr_ids",
        [](SamplingResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_nbr_res_nbr_ids(res));
        },
        py::return_value_policy::reference);

  m.def("get_nbr_res_edge_ids",
      [](SamplingResponse* res) {
        ImportNumpy();
        return py::reinterpret_steal<py::object>(
          get_nbr_res_edge_ids(res));
      },
      py::return_value_policy::reference);

  m.def("get_nbr_res_degrees",
      [](SamplingResponse* res) {
        ImportNumpy();
        return py::reinterpret_steal<py::object>(
          get_nbr_res_degrees(res));
      },
      py::return_value_policy::reference);

  m.def("new_get_edge_req",
        &new_get_edge_req,
        py::return_value_policy::reference,
        py::arg("edge_type"),
        py::arg("strategy"),
        py::arg("batch_size"));

  m.def("del_get_edge_req",
        [](GetEdgesRequest* req) {
          del_get_edge_req(req);
        });

  m.def("new_get_edge_res",
        &new_get_edge_res,
        py::return_value_policy::reference);

  m.def("del_get_edge_res",
        [](GetEdgesResponse* res) {
          del_get_edge_res(res);
        });

  m.def("get_edge_src_id_res",
        [](GetEdgesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(get_edge_src_id_res(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_dst_id_res",
        [](GetEdgesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(get_edge_dst_id_res(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_edge_id_res",
        [](GetEdgesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(get_edge_edge_id_res(res));
        },
        py::return_value_policy::reference);

  // Edge property
  m.def("new_lookup_edges_req",
      &new_lookup_edges_req,
      py::return_value_policy::reference,
      py::arg("edge_type"));

  m.def("del_lookup_edges_req",
        [](LookupEdgesRequest* req) {
          del_lookup_edges_req(req);
        });

  m.def("set_lookup_edges_req",
      [](LookupEdgesRequest* req,
        py::object src_ids,
        py::object edge_ids) {
        ImportNumpy();
        set_lookup_edges_req(req, src_ids.ptr(), edge_ids.ptr());
      });

  m.def("new_lookup_edges_res",
        &new_lookup_edges_res,
        py::return_value_policy::reference);

  m.def("del_lookup_edges_res",
        [](LookupEdgesResponse* res) {
          del_lookup_edges_res(res);
        });

  m.def("get_edge_int_attr_res",
        [](LookupEdgesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_edge_int_attr_res(res));
       },
       py::return_value_policy::reference);

  m.def("get_edge_float_attr_res",
        [](LookupEdgesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_edge_float_attr_res(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_string_attr_res",
        [](LookupEdgesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_edge_string_attr_res(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_weights_res",
        [](LookupEdgesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_edge_weights_res(res));
        },
        py::return_value_policy::reference);

  m.def("get_edge_labels_res",
    [](LookupEdgesResponse* res) {
      ImportNumpy();
      return py::reinterpret_steal<py::object>(
        get_edge_labels_res(res));
    },
    py::return_value_policy::reference);

  // Nodes
  m.def("new_get_node_req",
      &new_get_node_req,
      py::return_value_policy::reference,
      py::arg("node_type"),
      py::arg("strategy"),
      py::arg("node_from"),
      py::arg("batch_size"));

  m.def("del_get_node_req",
        [](GetNodesRequest* req) {
          del_get_node_req(req);
        });

  m.def("new_get_node_res",
        &new_get_node_res,
        py::return_value_policy::reference);

  m.def("del_get_node_res",
        [](GetNodesResponse* res) {
          del_get_node_res(res);
        });

  m.def("get_node_node_id_res",
        [](GetNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(get_node_node_id_res(res));
        },
        py::return_value_policy::reference);

  // Node property
  m.def("new_lookup_nodes_req",
      &new_lookup_nodes_req,
      py::return_value_policy::reference,
      py::arg("node_type"));

  m.def("del_lookup_nodes_req",
        [](LookupNodesRequest* req) {
          del_lookup_nodes_req(req);
        });

  m.def("set_lookup_nodes_req",
      [](LookupNodesRequest* req,
        py::object node_ids) {
        ImportNumpy();
        set_lookup_nodes_req(req, node_ids.ptr());
      });

  m.def("new_lookup_nodes_res",
        &new_lookup_nodes_res,
        py::return_value_policy::reference);

  m.def("del_lookup_nodes_res",
        [](LookupNodesResponse* res) {
          del_lookup_nodes_res(res);
        });

  m.def("get_node_weights_res",
        [](LookupNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_weights_res(res));
        },
        py::return_value_policy::reference);

  m.def("get_node_labels_res",
        [](LookupNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_labels_res(res));
        },
        py::return_value_policy::reference);

  m.def("get_node_int_attr_res",
        [](LookupNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_int_attr_res(res));
        },
        py::return_value_policy::reference);

  m.def("get_node_float_attr_res",
        [](LookupNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_float_attr_res(res));
        },
        py::return_value_policy::reference);

  m.def("get_node_string_attr_res",
        [](LookupNodesResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_string_attr_res(res));
        },
        py::return_value_policy::reference);

  m.def("new_agg_nodes_req",
      &new_agg_nodes_req,
      py::return_value_policy::reference,
      py::arg("node_type"),
      py::arg("strategy"));

  m.def("del_agg_nodes_req",
        [](AggregatingRequest* req) {
          del_agg_nodes_req(req);
        });

  m.def("set_agg_nodes_req",
      [](AggregatingRequest* req,
        py::object node_ids,
        py::object segement_ids,
        int32_t num_segments) {
        ImportNumpy();
        set_agg_nodes_req(req, node_ids.ptr(), segement_ids.ptr(), num_segments);
      });

  m.def("new_agg_nodes_res",
        &new_agg_nodes_res,
        py::return_value_policy::reference);

  m.def("del_agg_nodes_res",
        [](AggregatingResponse* res) {
          del_agg_nodes_res(res);
        });

  m.def("get_node_agg_res",
        [](AggregatingResponse* res) {
          ImportNumpy();
          return py::reinterpret_steal<py::object>(
            get_node_agg_res(res));
        },
        py::return_value_policy::reference);

} //NOLINT [readability/fn_size]

