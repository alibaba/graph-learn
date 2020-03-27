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

#include <fstream>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/graph/graph_store.h"
#include "graphlearn/core/graph/storage/graph_storage.h"
#include "graphlearn/core/graph/storage/node_storage.h"
#include "graphlearn/core/io/element_value.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/platform/env.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class GraphStoreTest : public ::testing::Test {
public:
  GraphStoreTest() {
    InitGoogleLogging();
  }
  ~GraphStoreTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
  }

  void TearDown() override {
  }

  void GenEdgeTestData(const char* file_name, int32_t format) {
    SideInfo info;
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
                        "%d\t%d\t%f\t%d\t%d:%f:%c\n",
                        i, i, float(i), i, i, float(i), char(i % 26 + 'A'));
      } else if (info.IsWeighted() && info.IsLabeled()) {
        size = snprintf(buffer, sizeof(buffer),
                        "%d\t%d\t%f\t%d\n", i, i, float(i), i);
      } else if (info.IsWeighted() && info.IsAttributed()) {
        size = snprintf(buffer, sizeof(buffer),
                        "%d\t%d\t%f\t%d:%f:%c\n",
                        i, i, float(i), i, float(i), char(i % 26 + 'A'));
      } else if (info.IsLabeled() && info.IsAttributed()) {
        size = snprintf(buffer, sizeof(buffer),
                        "%d\t%d\t%d\t%d:%f:%c\n",
                        i, i, i, i, float(i), char(i % 26 + 'A'));
      } else if (info.IsWeighted()) {
        size = snprintf(buffer, sizeof(buffer), "%d\t%d\t%f\n", i, i, float(i));
      } else if (info.IsLabeled()) {
        size = snprintf(buffer, sizeof(buffer), "%d\t%d\t%d\n", i, i, i);
      } else {
        size = snprintf(buffer, sizeof(buffer),
                        "%d\t%d\t%d:%f:%c\n",
                        i, i, i, float(i), char(i % 26 + 'A'));
      }
      out.write(buffer, size);
    }
    out.close();
  }

  void GenNodeTestData(const char* file_name, int32_t format) {
    SideInfo info;
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
                        "%d\t%f\t%d\t%d:%f:%c\n",
                        i, float(i), i, i, float(i), char(i % 26 + 'A'));
      } else if (info.IsWeighted() && info.IsLabeled()) {
        size = snprintf(buffer, sizeof(buffer),
                        "%d\t%f\t%d\n", i, float(i), i);
      } else if (info.IsWeighted() && info.IsAttributed()) {
        size = snprintf(buffer, sizeof(buffer),
                        "%d\t%f\t%d:%f:%c\n",
                        i, float(i), i, float(i), char(i % 26 + 'A'));
      } else if (info.IsLabeled() && info.IsAttributed()) {
        size = snprintf(buffer, sizeof(buffer),
                        "%d\t%d\t%d:%f:%c\n",
                        i, i, i, float(i), char(i % 26 + 'A'));
      } else if (info.IsWeighted()) {
        size = snprintf(buffer, sizeof(buffer), "%d\t%f\n", i, float(i));
      } else if (info.IsLabeled()) {
        size = snprintf(buffer, sizeof(buffer), "%d\t%d\n", i, i);
      } else {
        size = snprintf(buffer, sizeof(buffer),
                        "%d\t%d:%f:%c\n",
                        i, i, float(i), char(i % 26 + 'A'));
      }
      out.write(buffer, size);
    }
    out.close();
  }

  void GenEdgeSource(EdgeSource* source, int32_t format,
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
    if (format & kAttributed) {
      source->delimiter = ":";
      source->types = {DataType::kInt32, DataType::kFloat, DataType::kString};
      source->hash_buckets = {0 ,0, 0};
    }
  }

  void GenNodeSource(NodeSource* source, int32_t format,
                     const std::string& file_name,
                     const std::string& node_type) {
    source->path = file_name;
    source->id_type = node_type;
    source->format = format;
    source->ignore_invalid = false;
    if (format & kAttributed) {
      source->delimiter = ":";
      source->types = {DataType::kInt32, DataType::kFloat, DataType::kString};
      source->hash_buckets = {0 ,0, 0};
    }
  }

  void Check(EdgeValue& value, int32_t index, const SideInfo* info) {
    EXPECT_EQ(value.src_id, index);
    EXPECT_EQ(value.dst_id, index);
    if (info->IsWeighted()) {
      EXPECT_FLOAT_EQ(value.weight, float(index));
    }
    if (info->IsLabeled()) {
      EXPECT_EQ(value.label, index);
    }
    if (info->IsAttributed()) {
      EXPECT_EQ(value.i_attrs[0], index);
      EXPECT_FLOAT_EQ(value.f_attrs[0], float(index));
      EXPECT_EQ(value.s_attrs[0].length(), 1);
      EXPECT_EQ(value.s_attrs[0][0], char('A' + index % 26));
    }
  }

  void TestGraph(GraphStore* store) {
    Graph* graph = store->GetGraph("click");
    EXPECT_TRUE(graph != nullptr);
    GraphStorage* storage = graph->GetLocalStorage();
    EXPECT_EQ(storage->GetEdgeCount(), 100);
    EXPECT_EQ(storage->GetAllSrcIds()->size(), 100);
    EXPECT_EQ(storage->GetAllDstIds()->size(), 100);
    const SideInfo* info = storage->GetSideInfo();
    EXPECT_EQ(info->format, int32_t(kWeighted));
    EXPECT_EQ(info->i_num, 0);
    EXPECT_EQ(info->f_num, 0);
    EXPECT_EQ(info->s_num, 0);
    EXPECT_EQ(info->type, "click");
    EXPECT_EQ(info->src_type, "user");
    EXPECT_EQ(info->dst_type, "item");

    graph = store->GetGraph("buy");
    EXPECT_TRUE(graph != nullptr);
    storage = graph->GetLocalStorage();
    EXPECT_EQ(storage->GetEdgeCount(), 100);
    EXPECT_EQ(storage->GetAllSrcIds()->size(), 100);
    EXPECT_EQ(storage->GetAllDstIds()->size(), 100);
    info = storage->GetSideInfo();
    EXPECT_EQ(info->format, int32_t(kLabeled));
    EXPECT_EQ(info->i_num, 0);
    EXPECT_EQ(info->f_num, 0);
    EXPECT_EQ(info->s_num, 0);
    EXPECT_EQ(info->type, "buy");
    EXPECT_EQ(info->src_type, "user");
    EXPECT_EQ(info->dst_type, "item");

    graph = store->GetGraph("watch");
    EXPECT_TRUE(graph != nullptr);
    storage = graph->GetLocalStorage();
    EXPECT_EQ(storage->GetEdgeCount(), 100);
    EXPECT_EQ(storage->GetAllSrcIds()->size(), 100);
    EXPECT_EQ(storage->GetAllDstIds()->size(), 100);
    info = storage->GetSideInfo();
    EXPECT_EQ(info->format, int32_t(kAttributed));
    EXPECT_EQ(info->i_num, 1);
    EXPECT_EQ(info->f_num, 1);
    EXPECT_EQ(info->s_num, 1);
    EXPECT_EQ(info->type, "watch");
    EXPECT_EQ(info->src_type, "user");
    EXPECT_EQ(info->dst_type, "movie");
  }

  void TestNoder(GraphStore* store) {
    Noder* noder = store->GetNoder("user");
    EXPECT_TRUE(noder != nullptr);
    NodeStorage* storage = noder->GetLocalStorage();
    EXPECT_EQ(storage->Size(), 100);
    EXPECT_EQ(storage->GetIds()->size(), 100);
    EXPECT_EQ(storage->GetWeights()->size(), 100);
    EXPECT_EQ(storage->GetLabels()->size(), 0);
    EXPECT_EQ(storage->GetAttributes()->size(), 0);
    const SideInfo* info = storage->GetSideInfo();
    EXPECT_EQ(info->format, int32_t(kWeighted));
    EXPECT_EQ(info->i_num, 0);
    EXPECT_EQ(info->f_num, 0);
    EXPECT_EQ(info->s_num, 0);
    EXPECT_EQ(info->type, "user");

    noder = store->GetNoder("item");
    EXPECT_TRUE(noder != nullptr);
    storage = noder->GetLocalStorage();
    EXPECT_EQ(storage->Size(), 100);
    EXPECT_EQ(storage->GetIds()->size(), 100);
    EXPECT_EQ(storage->GetWeights()->size(), 0);
    EXPECT_EQ(storage->GetLabels()->size(), 100);
    EXPECT_EQ(storage->GetAttributes()->size(), 0);
    info = storage->GetSideInfo();
    EXPECT_EQ(info->format, int32_t(kLabeled));
    EXPECT_EQ(info->i_num, 0);
    EXPECT_EQ(info->f_num, 0);
    EXPECT_EQ(info->s_num, 0);
    EXPECT_EQ(info->type, "item");

    noder = store->GetNoder("movie");
    EXPECT_TRUE(noder != nullptr);
    storage = noder->GetLocalStorage();
    EXPECT_EQ(storage->Size(), 100);
    EXPECT_EQ(storage->GetIds()->size(), 100);
    EXPECT_EQ(storage->GetWeights()->size(), 0);
    EXPECT_EQ(storage->GetLabels()->size(), 0);
    EXPECT_EQ(storage->GetAttributes()->size(), 100);
    info = storage->GetSideInfo();
    EXPECT_EQ(info->format, int32_t(kAttributed));
    EXPECT_EQ(info->i_num, 1);
    EXPECT_EQ(info->f_num, 1);
    EXPECT_EQ(info->s_num, 1);
    EXPECT_EQ(info->type, "movie");
  }
};

TEST_F(GraphStoreTest, OnlyEdges) {
  const char* w_file = "w_edge_file";
  const char* l_file = "l_edge_file";
  const char* a_file = "a_edge_file";

  GenEdgeTestData(w_file, kWeighted);
  GenEdgeTestData(l_file, kLabeled);
  GenEdgeTestData(a_file, kAttributed);

  std::vector<EdgeSource> edge_source(3);
  GenEdgeSource(&edge_source[0], kWeighted, w_file, "click", "user", "item");
  GenEdgeSource(&edge_source[1], kLabeled, l_file, "buy", "user", "item");
  GenEdgeSource(&edge_source[2], kAttributed, a_file, "watch", "user", "movie");

  std::vector<NodeSource> node_source;
  GraphStore store(Env::Default());
  ::graphlearn::op::OperatorFactory::GetInstance().Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  TestGraph(&store);
}

TEST_F(GraphStoreTest, OnlyNodes) {
  const char* w_file = "w_node_file";
  const char* l_file = "l_node_file";
  const char* a_file = "a_node_file";

  GenNodeTestData(w_file, kWeighted);
  GenNodeTestData(l_file, kLabeled);
  GenNodeTestData(a_file, kAttributed);

  std::vector<NodeSource> node_source(3);
  GenNodeSource(&node_source[0], kWeighted, w_file, "user");
  GenNodeSource(&node_source[1], kLabeled, l_file, "item");
  GenNodeSource(&node_source[2], kAttributed, a_file, "movie");

  std::vector<EdgeSource> edge_source;
  GraphStore store(Env::Default());
  ::graphlearn::op::OperatorFactory::GetInstance().Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  TestNoder(&store);
}

TEST_F(GraphStoreTest, EdgesAndNodes) {
  std::vector<EdgeSource> edge_source(3);
  {
    const char* w_file = "w_edge_file";
    const char* l_file = "l_edge_file";
    const char* a_file = "a_edge_file";

    GenEdgeTestData(w_file, kWeighted);
    GenEdgeTestData(l_file, kLabeled);
    GenEdgeTestData(a_file, kAttributed);

    GenEdgeSource(&edge_source[0], kWeighted, w_file, "click", "user", "item");
    GenEdgeSource(&edge_source[1], kLabeled, l_file, "buy", "user", "item");
    GenEdgeSource(&edge_source[2], kAttributed, a_file, "watch", "user", "movie");
  }

  std::vector<NodeSource> node_source(3);
  {
    const char* w_file = "w_node_file";
    const char* l_file = "l_node_file";
    const char* a_file = "a_node_file";

    GenNodeTestData(w_file, kWeighted);
    GenNodeTestData(l_file, kLabeled);
    GenNodeTestData(a_file, kAttributed);

    GenNodeSource(&node_source[0], kWeighted, w_file, "user");
    GenNodeSource(&node_source[1], kLabeled, l_file, "item");
    GenNodeSource(&node_source[2], kAttributed, a_file, "movie");
  }

  GraphStore store(Env::Default());
  ::graphlearn::op::OperatorFactory::GetInstance().Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  TestGraph(&store);
  TestNoder(&store);
}
