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

#include <cstring>
#include <fstream>
#include <iostream>
#include "common/string/string_tool.h"
#include "core/io/element_value.h"
#include "include/server.h"
#include "include/config.h"
#include "include/client.h"

using namespace graphlearn;

void TestGetEdges(Client* client) {
  GetEdgesRequest req("click", "by_order", 64);
  GetEdgesResponse res;
  Status s = client->GetEdges(&req, &res);
  std::cout << "GetEdges: " << s.ToString() << std::endl;

  const int64_t* src_ids = res.SrcIds();
  const int64_t* dst_ids = res.DstIds();
  const int64_t* edge_ids = res.EdgeIds();
  int32_t size = res.Size();
  std::cout << "Size: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << src_ids[i] << ", " << dst_ids[i] << ", " << edge_ids[i] << std::endl;
  }
}

void TestGetNodes(Client* client) {
  GetNodesRequest req("user", "by_order", NodeFrom::kNode, 64);
  GetNodesResponse res;
  Status s = client->GetNodes(&req, &res);
  std::cout << "GetNodes: " << s.ToString() << std::endl;

  const int64_t* node_ids = res.NodeIds();
  int32_t size = res.Size();
  std::cout << "Size: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << node_ids[i] << std::endl;
  }
}

void TestLookupNodes(Client* client) {
  LookupNodesRequest req("movie");
  int64_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  req.Set(ids, 10);

  LookupNodesResponse res;
  Status s = client->LookupNodes(&req, &res);
  std::cout << "LookupNodes: " << s.ToString() << std::endl;

  int32_t size = res.Size();
  int32_t format = res.Format();
  std::cout << "Size: " << size << ", format: " << format << std::endl;
  if (res.Format() & io::DataFormat::kWeighted) {
    const float* weights = res.Weights();
    std::cout << "weights: ";
    for (int32_t i = 0; i < size; ++i) {
      std::cout << weights[i] << ' ';
    }
    std::cout << std::endl;
  }
  if (res.Format() & io::DataFormat::kLabeled) {
    const int32_t* labels = res.Labels();
    std::cout << "labels: ";
    for (int32_t i = 0; i < size; ++i) {
      std::cout << labels[i] << ' ';
    }
    std::cout << std::endl;
  }
  if (res.Format() & io::DataFormat::kAttributed) {
    if (res.IntAttrNum() > 0) {
      int32_t int_num = res.IntAttrNum();
      const int64_t* ints = res.IntAttrs();
      std::cout << "ints: ";
      for (int32_t i = 0; i < size * int_num; ++i) {
        std::cout << ints[i] << ' ';
      }
      std::cout << std::endl;
    }
    if (res.FloatAttrNum() > 0) {
      int32_t float_num = res.FloatAttrNum();
      const float* floats = res.FloatAttrs();
      std::cout << "floats: ";
      for (int32_t i = 0; i < size * float_num; ++i) {
        std::cout << floats[i] << ' ';
      }
      std::cout << std::endl;
    }
    if (res.StringAttrNum() > 0) {
      int32_t str_num = res.StringAttrNum();
      const std::string* const* strs = res.StringAttrs();
      std::cout << "strings: ";
      for (int32_t i = 0; i < size * str_num; ++i) {
        std::cout << *(strs[i]) << ' ';
      }
      std::cout << std::endl;
    }
  }
}

void TestSumAggregateNodes(Client* client) {
  AggregatingRequest req("movie", "SumAggregator");
  int64_t ids[10] = {0,  1, 2,  3, 4, 5,  6, 7, 8, 9};
  int32_t segment_ids[10] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
  int32_t num_segments = 4;
  req.Set(ids, segment_ids, 10, 4);

  AggregatingResponse res;
  Status s = client->Aggregating(&req, &res);
  std::cout << "SumAggregateNodes: " << s.ToString() << std::endl;

  int32_t size = res.NumSegments();

  if (res.EmbeddingDim() > 0) {
    int32_t float_num = res.EmbeddingDim();
    const float* floats = res.Embeddings();
    std::cout << "floats: ";
    for (int32_t i = 0; i < size * float_num; ++i) {
      std::cout << floats[i] << ' ';
    }
    std::cout << std::endl;
  }
}

void TestRandomSampleNeighbors(Client* client) {
  SamplingRequest req("click", "RandomSampler", 3);
  int64_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  req.Set(ids, 10);

  SamplingResponse res;
  Status s = client->Sampling(&req, &res);
  std::cout << "SampleNeighbors: " << s.ToString() << std::endl;

  const int64_t* nbrs = res.GetNeighborIds();
  int32_t size = res.GetShape().size;
  std::cout << "TotalNeighborCount: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << nbrs[i] << std::endl;
  }
}

void GenEdgeSource(io::EdgeSource* source, int32_t format,
                   const std::string& file_name,
                   const std::string& edge_type,
                   const std::string& src_type,
                   const std::string& dst_type) {
  source->path = file_name;
  source->edge_type = edge_type;
  source->src_id_type = src_type;
  source->dst_id_type = dst_type;
  source->format = format;
  source->attr_info.ignore_invalid = false;
  if (format & io::kAttributed) {
    source->attr_info.delimiter = ":";
    source->attr_info.types = {DataType::kInt32, DataType::kInt32, DataType::kFloat, DataType::kString};
    source->attr_info.hash_buckets = {0, 0, 0, 0};
  }
}

void GenNodeSource(io::NodeSource* source, int32_t format,
                   const std::string& file_name,
                   const std::string& node_type) {
  source->path = file_name;
  source->id_type = node_type;
  source->format = format;
  source->attr_info.ignore_invalid = false;
  if (format & io::kAttributed) {
    source->attr_info.delimiter = ":";
    source->attr_info.types = {DataType::kInt32, DataType::kInt32, DataType::kFloat, DataType::kString};
    source->attr_info.hash_buckets = {0, 0, 0, 0};
  }
}

void GenEdgeTestData(const char* file_name, int32_t format) {
  io::SideInfo info;
  info.format = format;

  std::ofstream out(file_name);

  // write title
  if (info.IsWeighted() && info.IsLabeled() && info.IsAttributed()) {
    const char* title = "src_id:int64\tdst_id:int64\tedge_weight:float\tlabel:int32\tattribute:string\n";
    out.write(title, strlen(title));
  } else if (info.IsWeighted() && info.IsLabeled()) {
    const char* title = "src_id:int64\tdst_id:int64\tedge_weight:float\tlabel:int32\n";
    out.write(title, strlen(title));
  } else if (info.IsWeighted() && info.IsAttributed()) {
    const char* title = "src_id:int64\tdst_id:int64\tedge_weight:float\tattribute:string\n";
    out.write(title, strlen(title));
  } else if (info.IsLabeled() && info.IsAttributed()) {
    const char* title = "src_id:int64\tdst_id:int64\tlabel:int32\tattribute:string\n";
    out.write(title, strlen(title));
  } else if (info.IsWeighted()) {
    const char* title = "src_id:int64\tdst_id:int64\tedge_weight:float\n";
    out.write(title, strlen(title));
  } else if (info.IsLabeled()) {
    const char* title = "src_id:int64\tdst_id:int64\tlabel:int32\n";
    out.write(title, strlen(title));
  } else {
    const char* title = "src_id:int64\tdst_id:int64\tattribute:string\n";
    out.write(title, strlen(title));
  }

  // write data
  int size = 0;
  char buffer[64];
  for (int32_t i = 0; i < 100; ++i) {
    if (info.IsWeighted() && info.IsLabeled() && info.IsAttributed()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%d\t%f\t%d\t%d:%d:%f:%c\n",
                      i, i, float(i), i, i, i*10, float(i), char(i % 26 + 'A'));
    } else if (info.IsWeighted() && info.IsLabeled()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%d\t%f\t%d\n", i, i, float(i), i);
    } else if (info.IsWeighted() && info.IsAttributed()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%d\t%f\t%d:%d:%f:%c\n",
                      i, i, float(i), i, i*10, float(i), char(i % 26 + 'A'));
    } else if (info.IsLabeled() && info.IsAttributed()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%d\t%d\t%d:%d:%f:%c\n",
                      i, i, i, i, i*10, float(i), char(i % 26 + 'A'));
    } else if (info.IsWeighted()) {
      size = snprintf(buffer, sizeof(buffer), "%d\t%d\t%f\n", i, i, float(i));
    } else if (info.IsLabeled()) {
      size = snprintf(buffer, sizeof(buffer), "%d\t%d\t%d\n", i, i, i);
    } else {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%d\t%d:%d:%f:%c\n",
                      i, i, i, i*10, float(i), char(i % 26 + 'A'));
    }
    out.write(buffer, size);
  }
  out.close();
}

void GenNodeTestData(const char* file_name, int32_t format) {
  io::SideInfo info;
  info.format = format;

  std::ofstream out(file_name);

  // write title
  if (info.IsWeighted() && info.IsLabeled() && info.IsAttributed()) {
    const char* title = "node_id:int64\tnode_weight:float\tlabel:int32\tattribute:string\n";
    out.write(title, strlen(title));
  } else if (info.IsWeighted() && info.IsLabeled()) {
    const char* title = "node_id:int64\tnode_weight:float\tlabel:int32\n";
    out.write(title, strlen(title));
  } else if (info.IsWeighted() && info.IsAttributed()) {
    const char* title = "node_id:int64\tnode_weight:float\tattribute:string\n";
    out.write(title, strlen(title));
  } else if (info.IsLabeled() && info.IsAttributed()) {
    const char* title = "node_id:int64\tlabel:int32\tattribute:string\n";
    out.write(title, strlen(title));
  } else if (info.IsWeighted()) {
    const char* title = "node_id:int64\tnode_weight:float\n";
    out.write(title, strlen(title));
  } else if (info.IsLabeled()) {
    const char* title = "node_id:int64\tlabel:int32\n";
    out.write(title, strlen(title));
  } else {
    const char* title = "node_id:int64\tattribute:string\n";
    out.write(title, strlen(title));
  }

  // write data
  int size = 0;
  char buffer[64];
  for (int32_t i = 0; i < 100; ++i) {
    if (info.IsWeighted() && info.IsLabeled() && info.IsAttributed()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%f\t%d\t%d:%d:%f:%c\n",
                      i, float(i), i, i, i*10, float(i), char(i % 26 + 'A'));
    } else if (info.IsWeighted() && info.IsLabeled()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%f\t%d\n", i, float(i), i);
    } else if (info.IsWeighted() && info.IsAttributed()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%f\t%d:%d:%f:%c\n",
                      i, float(i), i, i*10, float(i), char(i % 26 + 'A'));
    } else if (info.IsLabeled() && info.IsAttributed()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%d\t%d:%d:%f:%c\n",
                      i, i, i, i*10, float(i), char(i % 26 + 'A'));
    } else if (info.IsWeighted()) {
      size = snprintf(buffer, sizeof(buffer), "%d\t%f\n", i, float(i));
    } else if (info.IsLabeled()) {
      size = snprintf(buffer, sizeof(buffer), "%d\t%d\n", i, i);
    } else {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%d:%d:%f:%c\n",
                      i, i, i*10, float(i), char(i % 26 + 'A'));
    }
    out.write(buffer, size);
  }
  out.close();
}

int main(int argc, char** argv) {
  // ::system("mkdir -p ./tracker");

  int32_t server_id = 0;
  int32_t server_count = 1;
  std::string server_hosts = "";
  if (argc == 1) {
  } else if (argc > 3) {
    server_id = argv[1][0] - '0';
    server_count = argv[2][0] - '0';
    server_hosts = argv[3];
  } else {
    std::cout << "./dist_in_memory_test [server_id] [server_count] [server_hosts]" << std::endl;
    std::cout << "e.g. ./dist_in_memory_test 0 2 127.0.0.1:8888,127.0.0.1:8889" << std::endl;
    return -1;
  }

  SetGlobalFlagDeployMode(kWorker);

  SetGlobalFlagClientId(server_id);
  SetGlobalFlagClientCount(server_count);
  SetGlobalFlagServerCount(server_count);

  SetGlobalFlagTrackerMode(kRpc);
  // SetGlobalFlagTracker("./tracker");
  SetGlobalFlagServerHosts(server_hosts);

  LiteString s(server_hosts);
  std::string host = strings::Split(s, ",")[server_id];

  Server* server = NewServer(server_id, server_count, host, "./tracker");
  server->Start();
  std::cout << server_id << "th in " << server_count << " server started" << std::endl;

  GenEdgeTestData("weighted_edge_file", io::kWeighted);
  GenEdgeTestData("labeled_edge_file", io::kLabeled);
  GenEdgeTestData("attributed_edge_file", io::kAttributed);

  std::vector<io::EdgeSource> edges(3);
  GenEdgeSource(&edges[0], io::kWeighted, "weighted_edge_file", "click", "user", "item");
  GenEdgeSource(&edges[1], io::kLabeled, "labeled_edge_file", "buy", "user", "item");
  GenEdgeSource(&edges[2], io::kAttributed, "attributed_edge_file", "watch", "user", "movie");

  GenNodeTestData("weighted_node_file", io::kWeighted);
  GenNodeTestData("labeled_node_file", io::kLabeled);
  GenNodeTestData("attributed_node_file", io::kAttributed);

  std::vector<io::NodeSource> nodes(3);
  GenNodeSource(&nodes[0], io::kWeighted, "weighted_node_file", "user");
  GenNodeSource(&nodes[1], io::kLabeled, "labeled_node_file", "item");
  GenNodeSource(&nodes[2], io::kAttributed, "attributed_node_file", "movie");

  server->Init(edges, nodes);
  std::cout << "Inited" << std::endl;

  Client* client = NewInMemoryClient();

  TestGetEdges(client);
  TestGetNodes(client);
  TestLookupNodes(client);
  TestSumAggregateNodes(client);
  TestRandomSampleNeighbors(client);

  client->Stop();
  std::cout << "InMemory Client Stopped" << std::endl;
  delete client;

  server->Stop();
  std::cout << "Stopped" << std::endl;
  delete server;

  return 0;
}
