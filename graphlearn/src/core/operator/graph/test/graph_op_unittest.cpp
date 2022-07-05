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
#include "common/base/errors.h"
#include "common/base/log.h"
#include "core/graph/graph_store.h"
#include "core/io/element_value.h"
#include "core/operator/op_factory.h"
#include "include/config.h"
#include "platform/env.h"
#include "gtest/gtest.h"

using namespace graphlearn;      // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]
using namespace graphlearn::op;  // NOLINT [build/namespaces]

class GraphOpTest : public ::testing::Test {
public:
  GraphOpTest() {
    InitGoogleLogging();
  }
  ~GraphOpTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
    SetGlobalFlagInterThreadNum(1);
    SetGlobalFlagIntraThreadNum(1);
    SetGlobalFlagShuffleBufferSize(30);

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
    source->attr_info.ignore_invalid = false;
    if (format & kAttributed) {
      source->attr_info.delimiter = ":";
      source->attr_info.types = {DataType::kInt32, DataType::kFloat, DataType::kString};
      source->attr_info.hash_buckets = {0 ,0, 0};
    }
  }

  void GenNodeSource(NodeSource* source, int32_t format,
                     const std::string& file_name,
                     const std::string& node_type) {
    source->path = file_name;
    source->id_type = node_type;
    source->format = format;
    source->attr_info.ignore_invalid = false;
    if (format & kAttributed) {
      source->attr_info.delimiter = ":";
      source->attr_info.types = {DataType::kInt32, DataType::kFloat, DataType::kString};
      source->attr_info.hash_buckets = {0 ,0, 0};
    }
  }

protected:
  std::unordered_set<int64_t> id_set_;
};

TEST_F(GraphOpTest, EdgeGetter) {
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
  ::graphlearn::op::OpFactory::GetInstance()->Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  std::string edge_types[3] = {"click", "buy", "watch"};
  int32_t batch_size = 12;
  for (int32_t edge_type = 0; edge_type < 3; ++edge_type) {
    for (int32_t index = 0; index < 10; ++index) {
      GetEdgesRequest* req = new GetEdgesRequest(
        edge_types[edge_type], "by_order", batch_size);
      GetEdgesResponse* res = new GetEdgesResponse();

      Operator* op = OpFactory::GetInstance()->Create(req->Name());
      EXPECT_TRUE(op != nullptr);

      // 12 ids in the first 8 responses
      // 100-12*8=4 ids in the 9th
      // and then OutOfRange
      Status s = op->Process(req, res);
      if (index < 9) {
        EXPECT_TRUE(s.ok());
      } else {
        EXPECT_TRUE(error::IsOutOfRange(s));
        break;
      }

      if (index < 8) {
        EXPECT_EQ(res->Size(), batch_size);
      } else {
        EXPECT_EQ(res->Size(), 4);
      }

      const int64_t* src_ids = res->SrcIds();
      const int64_t* dst_ids = res->DstIds();
      for (int32_t i = 0; i < res->Size(); ++i) {
        EXPECT_EQ(src_ids[i], dst_ids[i]);
        EXPECT_TRUE(id_set_.find(src_ids[i]) != id_set_.end());
      }

      delete res;
      delete req;
    }


  }
}

TEST_F(GraphOpTest, NodeGetter) {
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
  ::graphlearn::op::OpFactory::GetInstance()->Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  std::string node_types[3] = {"user", "item", "movie"};
  int32_t batch_size = 12;
  for (int32_t node_type = 0; node_type < 3; ++node_type) {
    for (int32_t index = 0; index < 10; ++index) {
      GetNodesRequest* req = new GetNodesRequest(
        node_types[node_type], "by_order", NodeFrom::kNode, batch_size);
      GetNodesResponse* res = new GetNodesResponse();

      Operator* op = OpFactory::GetInstance()->Create(req->Name());
      EXPECT_TRUE(op != nullptr);

      // 12 ids in the first 8 responses
      // 100-12*8=4 ids in the 9th
      // and then OutOfRange
      Status s = op->Process(req, res);
      if (index < 9) {
        EXPECT_TRUE(s.ok());
      } else {
        EXPECT_TRUE(error::IsOutOfRange(s));
        break;
      }

      if (index < 8) {
        EXPECT_EQ(res->Size(), batch_size);
      } else {
        EXPECT_EQ(res->Size(), 4);
      }

      const int64_t* ids = res->NodeIds();
      for (int32_t i = 0; i < res->Size(); ++i) {
        EXPECT_TRUE(id_set_.find(ids[i]) != id_set_.end());
      }

      delete res;
      delete req;
    }
  }
}

TEST_F(GraphOpTest, ShuffledNodeGetter) {
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
  ::graphlearn::op::OpFactory::GetInstance()->Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  std::string node_types[3] = {"user", "item", "movie"};
  int32_t batch_size = 12;
  for (int32_t node_type = 0; node_type < 3; ++node_type) {
    for (int32_t index = 0; index < 10; ++index) {
      GetNodesRequest* req = new GetNodesRequest(
        node_types[node_type], "shuffle", NodeFrom::kNode, batch_size);
      GetNodesResponse* res = new GetNodesResponse();

      Operator* op = OpFactory::GetInstance()->Create(req->Name());
      EXPECT_TRUE(op != nullptr);

      // 12 ids in the first 8 responses
      // 100-12*8=4 ids in the 9th
      // and then OutOfRange
      Status s = op->Process(req, res);
      if (index < 9) {
        EXPECT_TRUE(s.ok());
      } else {
        EXPECT_TRUE(error::IsOutOfRange(s));
        break;
      }

      if (index < 8) {
        EXPECT_EQ(res->Size(), batch_size);
      } else {
        EXPECT_EQ(res->Size(), 4);
      }

      const int64_t* ids = res->NodeIds();
      for (int32_t i = 0; i < res->Size(); ++i) {
        EXPECT_TRUE(id_set_.find(ids[i]) != id_set_.end());
      }

      delete res;
      delete req;
    }
  }
}

TEST_F(GraphOpTest, NodeGetterFromEdgeSrc) {
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
  ::graphlearn::op::OpFactory::GetInstance()->Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  std::string edge_types[3] = {"click", "buy", "watch"};
  int32_t batch_size = 12;
  // get node from edge source node.
  for (int32_t edge_type = 0; edge_type < 3; ++edge_type) {
    for (int32_t index = 0; index < 10; ++index) {
      GetNodesRequest* req = new GetNodesRequest(
        edge_types[edge_type], "by_order", NodeFrom::kEdgeSrc, batch_size);
      GetNodesResponse* res = new GetNodesResponse();

      Operator* op = OpFactory::GetInstance()->Create(req->Name());
      EXPECT_TRUE(op != nullptr);

      // 12 ids in the first 8 responses
      // 100-12*8=4 ids in the 9th
      // and then OutOfRange
      Status s = op->Process(req, res);
      if (index < 9) {
        EXPECT_TRUE(s.ok());
      } else {
        EXPECT_TRUE(error::IsOutOfRange(s));
        break;
      }

      if (index < 8) {
        EXPECT_EQ(res->Size(), batch_size);
      } else {
        EXPECT_EQ(res->Size(), 4);
      }

      const int64_t* ids = res->NodeIds();
      for (int32_t i = 0; i < res->Size(); ++i) {
        EXPECT_TRUE(id_set_.find(ids[i]) != id_set_.end());
      }

      delete res;
      delete req;
    }
  }
  // get node from edge destination node.
  for (int32_t edge_type = 0; edge_type < 3; ++edge_type) {
    for (int32_t index = 0; index < 10; ++index) {
      GetNodesRequest* req = new GetNodesRequest(
        edge_types[edge_type], "by_order", NodeFrom::kEdgeDst, batch_size);
      GetNodesResponse* res = new GetNodesResponse();

      Operator* op = OpFactory::GetInstance()->Create(req->Name());
      EXPECT_TRUE(op != nullptr);

      // 12 ids in the first 8 responses
      // 100-12*8=4 ids in the 9th
      // and then OutOfRange
      Status s = op->Process(req, res);
      if (index < 9) {
        EXPECT_TRUE(s.ok());
      } else {
        EXPECT_TRUE(error::IsOutOfRange(s));
        break;
      }

      if (index < 8) {
        EXPECT_EQ(res->Size(), batch_size);
      } else {
        EXPECT_EQ(res->Size(), 4);
      }

      const int64_t* ids = res->NodeIds();
      for (int32_t i = 0; i < res->Size(); ++i) {
        EXPECT_TRUE(id_set_.find(ids[i]) != id_set_.end());
      }

      delete res;
      delete req;
    }
  }
}

TEST_F(GraphOpTest, EdgeLookuper) {
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
  ::graphlearn::op::OpFactory::GetInstance()->Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  int32_t batch_size = 100;
  std::vector<int64_t> ids;
  for (int32_t i = 0; i < batch_size; ++i) {
    // here we hack the ids. In single thread, the edge ids are increased by order.
    ids.push_back(i);
  }

  {
    LookupEdgesRequest* req = new LookupEdgesRequest("click");
    LookupEdgesResponse* res = new LookupEdgesResponse();
    req->Set(ids.data(), ids.data(), batch_size);

    Operator* op = OpFactory::GetInstance()->Create(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), 100);
    EXPECT_EQ(res->Format(), int(kWeighted));
    EXPECT_EQ(res->IntAttrNum(), 0);
    EXPECT_EQ(res->FloatAttrNum(), 0);
    EXPECT_EQ(res->StringAttrNum(), 0);

    const float* weights = res->Weights();
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_FLOAT_EQ(weights[i], float(i));
    }
    delete res;
    delete req;
  }
  {
    LookupEdgesRequest* req = new LookupEdgesRequest("buy");
    LookupEdgesResponse* res = new LookupEdgesResponse();
    req->Set(ids.data(), ids.data(), batch_size);

    Operator* op = OpFactory::GetInstance()->Create(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), 100);
    EXPECT_EQ(res->Format(), int(kLabeled));
    EXPECT_EQ(res->IntAttrNum(), 0);
    EXPECT_EQ(res->FloatAttrNum(), 0);
    EXPECT_EQ(res->StringAttrNum(), 0);

    const int32_t* labels = res->Labels();
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_EQ(labels[i], i);
    }
    delete res;
    delete req;
  }
  {
    LookupEdgesRequest* req = new LookupEdgesRequest("watch");
    LookupEdgesResponse* res = new LookupEdgesResponse();
    req->Set(ids.data(), ids.data(), batch_size);

    Operator* op = OpFactory::GetInstance()->Create(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), 100);
    EXPECT_EQ(res->Format(), int(kAttributed));
    EXPECT_EQ(res->IntAttrNum(), 1);
    EXPECT_EQ(res->FloatAttrNum(), 1);
    EXPECT_EQ(res->StringAttrNum(), 1);

    const int64_t* ints = res->IntAttrs();
    const float* floats = res->FloatAttrs();
    const std::string* const* strings = res->StringAttrs();
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_EQ(ints[i], i);
      EXPECT_FLOAT_EQ(floats[i], float(i));
      EXPECT_EQ(strings[i]->at(0), char('A' + i % 26));
    }
    delete res;
    delete req;
  }
}

TEST_F(GraphOpTest, NodeLookuper) {
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
  ::graphlearn::op::OpFactory::GetInstance()->Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  int32_t batch_size = 100;
  std::vector<int64_t> ids;
  for (int32_t i = 0; i < batch_size; ++i) {
    ids.push_back(i);
  }

  {
    LookupNodesRequest* req = new LookupNodesRequest("user");
    LookupNodesResponse* res = new LookupNodesResponse();
    req->Set(ids.data(), batch_size);

    Operator* op = OpFactory::GetInstance()->Create(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), 100);
    EXPECT_EQ(res->Format(), int(kWeighted));
    EXPECT_EQ(res->IntAttrNum(), 0);
    EXPECT_EQ(res->FloatAttrNum(), 0);
    EXPECT_EQ(res->StringAttrNum(), 0);

    const float* weights = res->Weights();
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_FLOAT_EQ(weights[i], float(i));
    }
    delete res;
    delete req;
  }
  {
    LookupNodesRequest* req = new LookupNodesRequest("item");
    LookupNodesResponse* res = new LookupNodesResponse();
    req->Set(ids.data(), batch_size);

    Operator* op = OpFactory::GetInstance()->Create(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), 100);
    EXPECT_EQ(res->Format(), int(kLabeled));
    EXPECT_EQ(res->IntAttrNum(), 0);
    EXPECT_EQ(res->FloatAttrNum(), 0);
    EXPECT_EQ(res->StringAttrNum(), 0);

    const int32_t* labels = res->Labels();
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_EQ(labels[i], i);
    }
    delete res;
    delete req;
  }
  {
    LookupNodesRequest* req = new LookupNodesRequest("movie");
    LookupNodesResponse* res = new LookupNodesResponse();
    req->Set(ids.data(), batch_size);

    Operator* op = OpFactory::GetInstance()->Create(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(res->Size(), 100);
    EXPECT_EQ(res->Format(), int(kAttributed));
    EXPECT_EQ(res->IntAttrNum(), 1);
    EXPECT_EQ(res->FloatAttrNum(), 1);
    EXPECT_EQ(res->StringAttrNum(), 1);

    const int64_t* ints = res->IntAttrs();
    const float* floats = res->FloatAttrs();
    const std::string* const* strings = res->StringAttrs();
    for (int32_t i = 0; i < res->Size(); ++i) {
      EXPECT_EQ(ints[i], i);
      EXPECT_FLOAT_EQ(floats[i], float(i));
      EXPECT_EQ(strings[i]->at(0), char('A' + i % 26));
    }
    delete res;
    delete req;
  }
}

TEST_F(GraphOpTest, NodeCountGetter) {
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
  ::graphlearn::op::OpFactory::GetInstance()->Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());
  s = store.Build(edge_source, node_source);
  EXPECT_TRUE(s.ok());
  
  for (int32_t i = 0; i < 3; ++i) {
    GetCountRequest* req = new GetCountRequest();
    GetCountResponse* res = new GetCountResponse();

    Operator* op = OpFactory::GetInstance()->Create(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    const int32_t* count = res->Count();
    EXPECT_EQ(count[0], 100);
    EXPECT_EQ(count[1], 100);
    EXPECT_EQ(count[2], 100);
    delete res;
    delete req;
  }
}

TEST_F(GraphOpTest, EdgeCountGetter) {
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
  ::graphlearn::op::OpFactory::GetInstance()->Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());
  s = store.Build(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  for (int32_t i = 0; i < 3; ++i) {
    GetCountRequest* req = new GetCountRequest();
    GetCountResponse* res = new GetCountResponse();

    Operator* op = OpFactory::GetInstance()->Create(req->Name());
    EXPECT_TRUE(op != nullptr);

    Status s = op->Process(req, res);
    EXPECT_TRUE(s.ok());
    const int32_t* count = res->Count();
    EXPECT_EQ(count[0], 100);
    EXPECT_EQ(count[1], 100);
    EXPECT_EQ(count[2], 100);
    delete res;
    delete req;
  }
}

TEST_F(GraphOpTest, DegreeGetter) {
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
  ::graphlearn::op::OpFactory::GetInstance()->Set(&store);

  Status s = store.Load(edge_source, node_source);
  EXPECT_TRUE(s.ok());

  int32_t batch_size = 64;
  std::vector<int64_t> ids;
  for (int32_t i = 0; i < batch_size; ++i) {
    ids.push_back(i);
  }
  GetDegreeRequest* req = new GetDegreeRequest("click", NodeFrom::kEdgeSrc);
  req->Set(ids.data(), batch_size);
  GetDegreeResponse* res = new GetDegreeResponse();

  Operator* op = OpFactory::GetInstance()->Create(req->Name());
  EXPECT_TRUE(op != nullptr);

  s = op->Process(req, res);
  EXPECT_TRUE(s.ok());
  const int32_t* degrees = res->GetDegrees();
  for (int32_t idx; idx < batch_size; ++idx) {
    EXPECT_EQ(degrees[idx], 1);
  }
  delete res;
  delete req;
}

