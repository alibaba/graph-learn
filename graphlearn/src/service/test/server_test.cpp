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
  source->option.name = "sort";
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
  source->option.name = "sort";
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
                      i, float(i), i, i%5, i*10, float(i), char(i % 26 + 'A'));
    } else if (info.IsWeighted() && info.IsLabeled()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%f\t%d\n", i, float(i), i);
    } else if (info.IsWeighted() && info.IsAttributed()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%f\t%d:%d:%f:%c\n",
                      i, float(i), i%5, i*10, float(i), char(i % 26 + 'A'));
    } else if (info.IsLabeled() && info.IsAttributed()) {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%d\t%d:%d:%f:%c\n",
                      i, i, i%5, i*10, float(i), char(i % 26 + 'A'));
    } else if (info.IsWeighted()) {
      size = snprintf(buffer, sizeof(buffer), "%d\t%f\n", i, float(i));
    } else if (info.IsLabeled()) {
      size = snprintf(buffer, sizeof(buffer), "%d\t%d\n", i, i);
    } else {
      size = snprintf(buffer, sizeof(buffer),
                      "%d\t%d:%d:%f:%c\n",
                      i, i%5, i*10, float(i), char(i % 26 + 'A'));
    }
    out.write(buffer, size);
  }
  out.close();
}

void GenSubGraphEdgeTestData(const char* file_name) {
  std::ofstream out(file_name);
  const char* title = "src_id:int64\tdst_id:int64\tedge_weight:float\n";
  out.write(title, strlen(title));

  int size = 0;
  char buffer[64];
  for (int32_t i = 0; i < 16; ++i) {
    size = snprintf(buffer, sizeof(buffer), "%d\t%d\t%f\n", i, i+2, float(i));
    out.write(buffer, size);
    size = snprintf(buffer, sizeof(buffer), "%d\t%d\t%f\n", i, i+3, float(i));
    out.write(buffer, size);
    size = snprintf(buffer, sizeof(buffer), "%d\t%d\t%f\n", i, i+5, float(i));
    out.write(buffer, size);
  }
  out.close();
}

void GenSubGraphNodeTestData(const char* file_name) {
  std::ofstream out(file_name);
  const char* title = "node_id:int64\tnode_weight:float\n";
  out.write(title, strlen(title));
  int size = 0;
  char buffer[64];
  for (int32_t i = 0; i < 20; ++i) {
    size = snprintf(buffer, sizeof(buffer), "%d\t%f\n", i, float(i));
    out.write(buffer, size);
  }
  out.close();
}


int main(int argc, char** argv) {
  int32_t server_id = 0;
  int32_t server_count = 1;
  std::string host;
  if (argc > 3) {
    server_id = argv[1][0] - '0';
    server_count = argv[2][0] - '0';
    SetGlobalFlagTrackerMode(kRpc);
    SetGlobalFlagServerHosts(argv[3]);

    LiteString s(argv[3]);
    host = strings::Split(s, ",")[server_id];
  } else if (argc > 2) {
    server_id = argv[1][0] - '0';
    server_count = argv[2][0] - '0';
    if (::system("mkdir -p ./tracker") != 0) {
      std::cerr << "cannot create tracker directory!" << ::std::endl;
    }
    SetGlobalFlagTracker("./tracker");
    SetGlobalFlagTrackerMode(kFileSystem);
  } else if (argc == 1) {
    if (::system("mkdir -p ./tracker") != 0) {
      std::cerr << "cannot create tracker directory!" << ::std::endl;
    }
    SetGlobalFlagTracker("./tracker");
    SetGlobalFlagTrackerMode(kFileSystem);
  } else {
    std::cout << "./server_test [server_id] [server_count] [hosts]" << std::endl;
    std::cout << "e.g. ./server_test 0 2 127.0.0.1:8888,127.0.0.1:8889" << std::endl;
    std::cout << "or ./server_test [server_id] [server_count]" << std::endl;
    std::cout << "e.g. ./server_test 0 2" << std::endl;
    return -1;
  }

  SetGlobalFlagDeployMode(kServer);

  Server* server = NewServer(server_id, server_count, host, "./tracker");
  server->Start();
  std::cout << server_id << "th in " << server_count << " server started" << std::endl;

  GenEdgeTestData("weighted_edge_file", io::kWeighted);
  GenEdgeTestData("labeled_edge_file", io::kLabeled);
  GenEdgeTestData("attributed_edge_file", io::kAttributed);
  GenSubGraphEdgeTestData("homo_edge_file");

  std::vector<io::EdgeSource> edges(4);
  GenEdgeSource(&edges[0], io::kWeighted, "weighted_edge_file", "click", "user", "item");
  GenEdgeSource(&edges[1], io::kLabeled, "labeled_edge_file", "buy", "user", "item");
  GenEdgeSource(&edges[2], io::kAttributed, "attributed_edge_file", "watch", "user", "movie");
  GenEdgeSource(&edges[3], io::kWeighted, "homo_edge_file", "r", "e", "e");

  GenNodeTestData("weighted_node_file", io::kWeighted);
  GenNodeTestData("labeled_node_file", io::kLabeled);
  GenNodeTestData("attributed_node_file", io::kWeighted|io::kAttributed);
  GenSubGraphNodeTestData("homo_node_file");

  std::vector<io::NodeSource> nodes(4);
  GenNodeSource(&nodes[0], io::kWeighted, "weighted_node_file", "user");
  GenNodeSource(&nodes[1], io::kLabeled, "labeled_node_file", "item");
  GenNodeSource(&nodes[2], io::kWeighted|io::kAttributed, "attributed_node_file", "movie");
  GenNodeSource(&nodes[3], io::kWeighted, "homo_node_file", "e");

  server->Init(edges, nodes);
  std::cout << "Inited" << std::endl;

  server->Stop();
  std::cout << "Stopped" << std::endl;
  delete server;

  return 0;
}
