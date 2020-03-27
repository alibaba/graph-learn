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
#include <unordered_set>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/graph/graph_store.h"
#include "graphlearn/include/aggregation_request.h"
#include "graphlearn/core/io/element_value.h"
#include "graphlearn/core/operator/operator_factory.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"
#include "gtest/gtest.h"

using namespace graphlearn;      // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]
using namespace graphlearn::op;  // NOLINT [build/namespaces]

class AggregationOpTest : public ::testing::Test {
public:
  AggregationOpTest() {
    InitGoogleLogging();
  }
  ~AggregationOpTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
    SetGlobalFlagInterThreadNum(1);
    SetGlobalFlagIntraThreadNum(1);

    for (int32_t i = 0; i < 100; ++i) {
      id_set_.insert(i);
    }
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

protected:
  std::unordered_set<int64_t> id_set_;
};

TEST_F(AggregationOpTest, Aggregator) {
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

  int32_t id_size = 10;
  std::vector<int64_t> ids;
  for (int32_t i = 0; i < id_size; ++i) {
    ids.push_back(i);
  }
  int32_t num_segments = 5;
  std::vector<int32_t> segments;
  for (int32_t j = 0; j < num_segments; ++j) {
    segments.push_back(j);
  }

  {
    AggregateNodesRequest* req = new AggregateNodesRequest("movie", "SumAggregator");
    AggregateNodesResponse* res = new AggregateNodesResponse();
    req->Set(ids.data(), id_size, segments.data(), num_segments);

    Operator* op = OperatorFactory::GetInstance().Lookup(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), num_segments);
    EXPECT_EQ(res->Format(), int(kAttributed));
    EXPECT_EQ(res->IntAttrNum(), 1);
    EXPECT_EQ(res->FloatAttrNum(), 1);
    EXPECT_EQ(res->StringAttrNum(), 1);

    const float* floats = res->FloatAttrs();
    int32_t reduced_float_attrs[5] = {0, 0, 3, 12, 30};
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_FLOAT_EQ(floats[i], reduced_float_attrs[i]);
    }
    delete res;
    delete req;
  }

  {
    AggregateNodesRequest* req = new AggregateNodesRequest("movie", "MeanAggregator");
    AggregateNodesResponse* res = new AggregateNodesResponse();
    req->Set(ids.data(), id_size, segments.data(), num_segments);

    Operator* op = OperatorFactory::GetInstance().Lookup(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), num_segments);
    EXPECT_EQ(res->Format(), int(kAttributed));
    EXPECT_EQ(res->IntAttrNum(), 1);
    EXPECT_EQ(res->FloatAttrNum(), 1);
    EXPECT_EQ(res->StringAttrNum(), 1);

    const float* floats = res->FloatAttrs();
    double reduced_float_attrs[5] = {0.0, 0.0, 3.0 / 2, 12.0 / 3 , 30.0 / 4};
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_FLOAT_EQ(floats[i], reduced_float_attrs[i]);
    }
    delete res;
    delete req;
  }

  {
    AggregateNodesRequest* req = new AggregateNodesRequest("movie", "MinAggregator");
    AggregateNodesResponse* res = new AggregateNodesResponse();
    req->Set(ids.data(), id_size, segments.data(), num_segments);

    Operator* op = OperatorFactory::GetInstance().Lookup(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), num_segments);
    EXPECT_EQ(res->Format(), int(kAttributed));
    EXPECT_EQ(res->IntAttrNum(), 1);
    EXPECT_EQ(res->FloatAttrNum(), 1);
    EXPECT_EQ(res->StringAttrNum(), 1);

    const float* floats = res->FloatAttrs();
    int32_t reduced_float_attrs[5] = {0, 0, 1, 3, 6 };
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_FLOAT_EQ(floats[i], reduced_float_attrs[i]);
    }
    delete res;
    delete req;
  }

  {
    AggregateNodesRequest* req = new AggregateNodesRequest("movie", "MaxAggregator");
    AggregateNodesResponse* res = new AggregateNodesResponse();
    req->Set(ids.data(), id_size, segments.data(), num_segments);

    Operator* op = OperatorFactory::GetInstance().Lookup(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), num_segments);
    EXPECT_EQ(res->Format(), int(kAttributed));
    EXPECT_EQ(res->IntAttrNum(), 1);
    EXPECT_EQ(res->FloatAttrNum(), 1);
    EXPECT_EQ(res->StringAttrNum(), 1);

    const float* floats = res->FloatAttrs();
    int32_t reduced_float_attrs[5] = {0, 0, 2, 5, 9};
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_FLOAT_EQ(floats[i], reduced_float_attrs[i]);
    }
    delete res;
    delete req;
  }

  {
    AggregateNodesRequest* req = new AggregateNodesRequest("movie", "ProdAggregator");
    AggregateNodesResponse* res = new AggregateNodesResponse();
    req->Set(ids.data(), id_size, segments.data(), num_segments);

    Operator* op = OperatorFactory::GetInstance().Lookup(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), num_segments);
    EXPECT_EQ(res->Format(), int(kAttributed));
    EXPECT_EQ(res->IntAttrNum(), 1);
    EXPECT_EQ(res->FloatAttrNum(), 1);
    EXPECT_EQ(res->StringAttrNum(), 1);

    const float* floats = res->FloatAttrs();
    int32_t reduced_float_attrs[5] = {0, 0, 2, 60, 3024};
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_FLOAT_EQ(floats[i], reduced_float_attrs[i]);
    }
    delete res;
    delete req;
  }
}
