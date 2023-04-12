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

#include "include/client.h"
#include "include/config.h"
#include "include/data_source.h"
#include "include/index_option.h"
#include "include/server.h"
#include "include/status.h"
#include "python/c/py_bind.h"

using namespace graphlearn;

void init_client_module(py::module&);

#ifdef OPEN_KNN
void init_contrib_module(py::module&);
#endif

PYBIND11_MODULE(pywrap_graphlearn, m) {
  m.doc() = "Python interface for graph-learn.";
  // setters
  m.def("set_default_neighbor_id", &SetGlobalFlagDefaultNeighborId);
  m.def("set_tracker_mode", &SetGlobalFlagTrackerMode);
  m.def("set_padding_mode", &SetGlobalFlagPaddingMode);
  m.def("set_storage_mode", &SetGlobalFlagStorageMode);
  m.def("set_default_int_attr", &SetGlobalFlagDefaultIntAttribute);
  m.def("set_default_float_attr", &SetGlobalFlagDefaultFloatAttribute);
  m.def("set_default_string_attr", &SetGlobalFlagDefaultStringAttribute);
  m.def("set_default_weight", &SetGlobalFlagDefaultWeight);
  m.def("set_default_label", &SetGlobalFlagDefaultLabel);
  m.def("set_default_timestamp", &SetGlobalFlagDefaultTimestamp);
  m.def("set_retry_times", &SetGlobalFlagRetryTimes);
  m.def("set_timeout", &SetGlobalFlagTimeout);
  m.def("set_inmemory_queuesize", &SetGlobalFlagInMemoryQueueSize);
  m.def("set_inner_threadnum", &SetGlobalFlagInterThreadNum);
  m.def("set_inter_threadnum", &SetGlobalFlagInterThreadNum);
  m.def("set_intra_threadnum", &SetGlobalFlagIntraThreadNum);
  m.def("set_datainit_batchsize", &SetGlobalFlagDataInitBatchSize);
  m.def("set_shuffle_buffer_size", &SetGlobalFlagShuffleBufferSize);
  m.def("set_rpc_message_max_size", &SetGlobalFlagRpcMessageMaxSize);
  m.def("set_deploy_mode", &SetGlobalFlagDeployMode);
  m.def("set_client_id", &SetGlobalFlagClientId);
  m.def("set_client_count", &SetGlobalFlagClientCount);
  m.def("set_server_count", &SetGlobalFlagServerCount);
  m.def("set_tracker", &SetGlobalFlagTracker);
  m.def("set_server_hosts", &SetGlobalFlagServerHosts);
  m.def("set_knn_metric", &SetGlobalFlagKnnMetric);
  m.def("set_tape_capacity", &SetGlobalFlagTapeCapacity);
  m.def("set_dataset_capacity", &SetGlobalFlagDatasetCapacity);
  m.def("set_ignore_invalid", &SetGlobalFlagIgnoreInvalid);
  m.def("set_sampler_retry_times", &SetGlobalFlagSamplingRetryTimes);
  m.def("set_field_delimiter", &SetGlobalFlagFieldDelimiter);
  m.def("set_default_full_nbr_num", &SetGlobalFlagDefaultFullNbrNum);
  m.def("set_local_node_cache_capacity", &SetGlobalFlagLocalNodeCacheCapacity);
  // For Actor
  m.def("set_enable_actor", &SetGlobalFlagEnableActor);
  m.def("set_actor_local_shard_count", &SetGlobalFlagActorLocalShardCount);

  // Constants
  m.attr("kOpName") = kOpName;
  m.attr("kNodeType") = kNodeType;
  m.attr("kEdgeType") = kEdgeType;
  m.attr("kType") = kType;
  m.attr("kSrcType") = kSrcType;
  m.attr("kDstType") = kDstType;
  m.attr("kSrcIds") = kSrcIds;
  m.attr("kDstIds") = kDstIds;
  m.attr("kNodeIds") = kNodeIds;
  m.attr("kEdgeIds") = kEdgeIds;
  m.attr("kNeighborCount") = kNeighborCount;
  m.attr("kNeighborIds") = kNeighborIds;
  m.attr("kBatchSize") = kBatchSize;
  m.attr("kIsSparse") = kIsSparse;
  m.attr("kStrategy") = kStrategy;
  m.attr("kDegreeKey") = kDegreeKey;
  m.attr("kWeightKey") = kWeightKey;
  m.attr("kLabelKey") = kLabelKey;
  m.attr("kIntAttrKey") = kIntAttrKey;
  m.attr("kFloatAttrKey") = kFloatAttrKey;
  m.attr("kStringAttrKey") = kStringAttrKey;
  m.attr("kSideInfo") = kSideInfo;
  m.attr("kDirection") = kDirection;
  m.attr("kSegmentIds") = kSegmentIds;
  m.attr("kNumSegments") = kNumSegments;
  m.attr("kSegments") = kSegments;
  m.attr("kDistances") = kDistances;
  m.attr("kRowIndices") = kRowIndices;
  m.attr("kColIndices") = kColIndices;
  m.attr("kSeedType") = kSeedType;
  m.attr("kNbrType") = kNbrType;
  m.attr("kCount") = kCount;
  m.attr("kBatchShare") = kBatchShare;
  m.attr("kUnique") = kUnique;
  m.attr("kIntCols") = kIntCols;
  m.attr("kIntProps") = kIntProps;
  m.attr("kFloatCols") = kFloatCols;
  m.attr("kFloatProps") = kFloatProps;
  m.attr("kStrCols") = kStrCols;
  m.attr("kStrProps") = kStrProps;
  m.attr("kFilterType") = kFilterType;
  m.attr("kFilterField") = kFilterField;
  m.attr("kFilterValues") = kFilterValues;
  m.attr("kDegrees") = kDegrees;
  m.attr("kEpoch") = kEpoch;
  m.attr("kNodeFrom") = kNodeFrom;
  m.attr("kNeedDist") = kNeedDist;
  m.attr("kDistToSrc") = kDistToSrc;
  m.attr("kDistToDst") = kDistToDst;

  // getters
  m.def("get_tracker_mode", &GetGlobalFlagTrackerMode);

  m.def("set_vineyard_graph_id", &SetGlobalFlagVineyardGraphID);
  m.def("set_vineyard_ipc_socket", &SetGlobalFlagVineyardIPCSocket);

  py::enum_<error::Code>(m, "ErrorCode")
    .value("OK", error::Code::OK)
    .value("CANCELLED", error::Code::CANCELLED)
    .value("UNKNOWN", error::Code::UNKNOWN)
    .value("INVALID_ARGUMENT", error::Code::INVALID_ARGUMENT)
    .value("DEADLINE_EXCEEDED", error::Code::DEADLINE_EXCEEDED)
    .value("NOT_FOUND", error::Code::NOT_FOUND)
    .value("ALREADY_EXISTS", error::Code::ALREADY_EXISTS)
    .value("PERMISSION_DENIED", error::Code::PERMISSION_DENIED)
    .value("UNAUTHENTICATED", error::Code::UNAUTHENTICATED)
    .value("RESOURCE_EXHAUSTED", error::Code::RESOURCE_EXHAUSTED)
    .value("FAILED_PRECONDITION", error::Code::FAILED_PRECONDITION)
    .value("ABORTED", error::Code::ABORTED)
    .value("OUT_OF_RANGE", error::Code::OUT_OF_RANGE)
    .value("UNIMPLEMENTED", error::Code::UNIMPLEMENTED)
    .value("INTERNAL", error::Code::INTERNAL)
    .value("UNAVAILABLE", error::Code::UNAVAILABLE)
    .value("DATA_LOSS", error::Code::DATA_LOSS)
    .value("REQUEST_STOP", error::Code::REQUEST_STOP);

  py::enum_<DataType>(m, "DataType")
    .value("INT32", DataType::kInt32)
    .value("INT64", DataType::kInt64)
    .value("FLOAT", DataType::kFloat)
    .value("DOUBLE", DataType::kDouble)
    .value("STRING", DataType::kString);

  py::enum_<DeployMode>(m, "DeployMode")
    .value("LOCAL", DeployMode::kLocal)
    .value("SERVER", DeployMode::kServer)
    .value("WORKER", DeployMode::kWorker);

  py::enum_<PartitionMode>(m, "PartitionMode")
    .value("NO_PARTITION", PartitionMode::kNoPartition)
    .value("BY_SOURCE_ID", PartitionMode::kByHash);

  py::enum_<PaddingMode>(m, "PaddingMode")
    .value("REPLICATE", PaddingMode::kReplicate)
    .value("CIRCULAR", PaddingMode::kCircular);

  py::enum_<TrackerMode>(m, "TrackerMode")
    .value("RPC", TrackerMode::kRpc)
    .value("FILE_SYSTEM", TrackerMode::kFileSystem);

  py::class_<IndexOption>(m, "IndexOption")
    .def(py::init<>())
    .def_readwrite("name", &IndexOption::name)
    .def_readwrite("index_type", &IndexOption::index_type)
    .def_readwrite("dimension", &IndexOption::dimension)
    .def_readwrite("nlist", &IndexOption::nlist)
    .def_readwrite("nprobe", &IndexOption::nprobe)
    .def_readwrite("m", &IndexOption::m);

  py::enum_<NodeFrom>(m, "NodeFrom")
    .value("EDGE_SRC", NodeFrom::kEdgeSrc)
    .value("EDGE_DST", NodeFrom::kEdgeDst)
    .value("NODE", NodeFrom::kNode);

  py::enum_<FilterType>(m, "FilterType")
    .value("OPERATOR_UNSPECIFIED", FilterType::kOperatorUnspecified)
    .value("EQUAL", FilterType::kEqual)
    .value("LARGER_THAN", FilterType::kLargerThan);

  py::enum_<FilterField>(m, "FilterField")
    .value("FIELD_UNSPECIFIED", FilterField::kFieldUnspecified)
    .value("ID", FilterField::kId)
    .value("TIMESTAMP", FilterField::kTimestamp);

  py::enum_<io::DataFormat>(m, "DataFormat")
    .value("DEFAULT", io::DataFormat::kDefault)
    .value("WEIGHTED", io::DataFormat::kWeighted)
    .value("LABELED", io::DataFormat::kLabeled)
    .value("ATTRIBUTED", io::DataFormat::kAttributed)
    .value("TIMESTAMPED", io::DataFormat::kTimestamped);

  py::enum_<io::Direction>(m, "Direction")
    .value("ORIGIN", io::Direction::kOrigin)
    .value("REVERSED", io::Direction::kReversed);

  py::class_<io::AttributeInfo>(m, "AttributeInfo")
    .def(py::init<>())
    .def_readwrite("delimiter", &io::AttributeInfo::delimiter)
    .def_readwrite("ignore_invalid", &io::AttributeInfo::ignore_invalid)
    .def("append_type", &io::AttributeInfo::AppendType)
    .def("append_hash_bucket", &io::AttributeInfo::AppendHashBucket);

  py::class_<io::NodeSource>(m, "NodeSource")
    .def(py::init<>())
    .def_readwrite("path", &io::NodeSource::path)
    .def_readwrite("id_type", &io::NodeSource::id_type)
    .def_readwrite("format", &io::NodeSource::format)
    .def_readwrite("attr_info", &io::NodeSource::attr_info)
    .def_readwrite("option", &io::NodeSource::option)
    .def_readwrite("view_type", &io::NodeSource::view_type)
    .def_readwrite("use_attrs", &io::NodeSource::use_attrs);

  py::class_<io::EdgeSource>(m, "EdgeSource")
    .def(py::init<>())
    .def_readwrite("path", &io::EdgeSource::path)
    .def_readwrite("edge_type", &io::EdgeSource::edge_type)
    .def_readwrite("src_id_type", &io::EdgeSource::src_id_type)
    .def_readwrite("dst_id_type", &io::EdgeSource::dst_id_type)
    .def_readwrite("format", &io::EdgeSource::format)
    .def_readwrite("direction", &io::EdgeSource::direction)
    .def_readwrite("attr_info", &io::EdgeSource::attr_info)
    .def_readwrite("option", &io::EdgeSource::option)
    .def_readwrite("view_type", &io::EdgeSource::view_type)
    .def_readwrite("use_attrs", &io::EdgeSource::use_attrs);

  py::class_<Status>(m, "Status")
    .def("ok", &Status::ok)
    .def("code", &Status::code)
    .def("message", &Status::msg)
    .def("to_string", &Status::ToString);

  py::class_<Server>(m, "Server")
    .def("start", &Server::Start)
    .def("init", &Server::Init)
    .def("stop", &Server::Stop)
    .def("stop_sampling", &Server::StopSampling)
    .def("get_stats", &Server::GetStats);

  m.def("server",
        &NewServer,
        py::return_value_policy::take_ownership,
        py::arg("server_id"),
        py::arg("server_count"),
        py::arg("server_host"),
        py::arg("tracker"));

  m.def("in_memory_client",
        &NewInMemoryClient,
        py::return_value_policy::take_ownership);

  m.def("rpc_client",
        &NewRpcClient,
        py::return_value_policy::take_ownership,
        py::arg("server_id") = -1,
        py::arg("client_own") = true);

  init_client_module(m);

#ifdef OPEN_KNN
  init_contrib_module(m);
#endif
} //NOLINT [readability/fn_size]
