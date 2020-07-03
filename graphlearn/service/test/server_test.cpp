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
#include <string>
#include "graphlearn/common/string/string_tool.h"
#include "graphlearn/core/io/element_value.h"
#include "graphlearn/include/server.h"
#include "graphlearn/include/config.h"

using namespace graphlearn;

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
  source->ignore_invalid = false;
  if (format & io::kAttributed) {
    source->delimiter = ":";
    source->types = {DataType::kInt32, DataType::kInt32, DataType::kFloat, DataType::kString};
    source->hash_buckets = {0, 0, 0, 0};
  }
}

void GenNodeSource(io::NodeSource* source, int32_t format,
                   const std::string& file_name,
                   const std::string& node_type) {
  source->path = file_name;
  source->id_type = node_type;
  source->format = format;
  source->ignore_invalid = false;
  if (format & io::kAttributed) {
    source->delimiter = ":";
    source->types = {DataType::kInt32, DataType::kInt32, DataType::kFloat, DataType::kString};
    source->hash_buckets = {0, 0, 0, 0};
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
  std::string hosts = "127.0.0.1:8888";
  if (argc == 1) {
  } else if (argc > 3) {
    server_id = argv[1][0] - '0';
    server_count = argv[2][0] - '0';
    hosts = argv[3];
  } else {
    std::cout << "./server_test [server_id] [server_count] [hosts]" << std::endl;
    std::cout << "e.g. ./server_test 0 2 127.0.0.1:8888,127.0.0.1:8889" << std::endl;
    return -1;
  }

  SetGlobalFlagDeployMode(1);
  SetGlobalFlagTrackerMode(0);
  SetGlobalFlagServerHosts(hosts);

  LiteString s(hosts);
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

  server->Stop();
  std::cout << "Stopped" << std::endl;
  delete server;

  return 0;
}
