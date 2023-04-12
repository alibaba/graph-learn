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

#include <iostream>
#include "include/client.h"
#include "include/config.h"

using namespace graphlearn;

void TestGetEdges(Client* client) {
  for (int32_t epoch = 0; epoch < 3; ++epoch) {
    for (int32_t iter = 0; iter < 100; ++iter) {
      GetEdgesRequest req("click", "by_order", 64, epoch);
      GetEdgesResponse res;
      Status s = client->GetEdges(&req, &res);
      std::cout << "GetEdges: " << s.ToString() << std::endl;

      if (!s.ok()) {
        std::cout << "OutOfRange, epoch:" << epoch << std::endl;
        break;
      }

      const int64_t* src_ids = res.SrcIds();
      const int64_t* dst_ids = res.DstIds();
      const int64_t* edge_ids = res.EdgeIds();
      int32_t size = res.Size();
      std::cout << "Size: " << size << std::endl;
      if (epoch == 0 && iter == 0) {
        for (int32_t i = 0; i < size; ++i) {
          std::cout << src_ids[i] << ", " << dst_ids[i] << ", " << edge_ids[i] << std::endl;
        }
      }
    }
  }
}

void TestGetNodes(Client* client) {
  for (int32_t epoch = 0; epoch < 3; ++epoch) {
    for (int32_t iter = 0; iter < 10; ++iter) {
      GetNodesRequest req("user", "by_order", NodeFrom::kNode, 64, epoch);
      GetNodesResponse res;
      Status s = client->GetNodes(&req, &res);
      std::cout << "GetNodes: " << s.ToString() << std::endl;

      if (!s.ok()) {
        std::cout << "OutOfRange, epoch:" << epoch << std::endl;
        break;
      }
      const int64_t* node_ids = res.NodeIds();
      int32_t size = res.Size();
      std::cout << "Size: " << size << std::endl;
      for (int32_t i = 0; i < size; ++i) {
        std::cout << node_ids[i] << std::endl;
      }
    }
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

void TestMeanAggregateNodes(Client* client) {
  AggregatingRequest req("movie", "MeanAggregator");
  int64_t ids[10] = {0,  1, 2,  3, 4, 5,  6, 7, 8, 9};
  int32_t segment_ids[10] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
  int32_t num_segments = 4;
  req.Set(ids, segment_ids, 10, 4);

  AggregatingResponse res;
  Status s = client->Aggregating(&req, &res);
  std::cout << "MeanAggregateNodes: " << s.ToString() << std::endl;

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

void TestMaxAggregateNodes(Client* client) {
  AggregatingRequest req("movie", "MaxAggregator");
  int64_t ids[10] = {0,  1, 2,  3, 4, 5,  6, 7, 8, 9};
  int32_t segment_ids[10] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
  int32_t num_segments = 4;
  req.Set(ids, segment_ids, 10, 4);

  AggregatingResponse res;
  Status s = client->Aggregating(&req, &res);
  std::cout << "MaxAggregateNodes: " << s.ToString() << std::endl;

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

void TestMinAggregateNodes(Client* client) {
  AggregatingRequest req("movie", "MinAggregator");
  int64_t ids[10] = {0,  1, 2,  3, 4, 5,  6, 7, 8, 9};
  int32_t segment_ids[10] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
  int32_t num_segments = 4;
  req.Set(ids, segment_ids, 10, 4);

  AggregatingResponse res;
  Status s = client->Aggregating(&req, &res);
  std::cout << "MinAggregateNodes: " << s.ToString() << std::endl;

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

void TestProdAggregateNodes(Client* client) {
  AggregatingRequest req("movie", "ProdAggregator");
  int64_t ids[10] = {0,  1, 2,  3, 4, 5,  6, 7, 8, 9};
  int32_t segment_ids[10] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
  int32_t num_segments = 4;
  req.Set(ids, segment_ids, 10, 4);

  AggregatingResponse res;
  Status s = client->Aggregating(&req, &res);
  std::cout << "ProdAggregateNodes: " << s.ToString() << std::endl;

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
  std::cout << "RandomSampleNeighbors: " << s.ToString() << std::endl;

  const int64_t* nbrs = res.GetNeighborIds();
  int32_t size = res.GetShape().size;
  std::cout << "TotalNeighborCount: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << nbrs[i] << std::endl;
  }
}

void TestRandomWithoutReplacementSampleNeighbors(Client* client) {
  SamplingRequest req("click", "RandomWithoutReplacementSampler", 3);
  int64_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  req.Set(ids, 10);

  SamplingResponse res;
  Status s = client->Sampling(&req, &res);
  std::cout << "RandomWithoutReplacementSampleNeighbors: " << s.ToString() << std::endl;

  const int64_t* nbrs = res.GetNeighborIds();
  int32_t size = res.GetShape().size;
  std::cout << "TotalNeighborCount: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << nbrs[i] << std::endl;
  }
}

void TestTopkSampleNeighbors(Client* client) {
  SamplingRequest req("click", "TopkSampler", 3);
  int64_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  req.Set(ids, 10);

  SamplingResponse res;
  Status s = client->Sampling(&req, &res);
  std::cout << "SampleTopkNeighbors: " << s.ToString() << std::endl;

  const int64_t* nbrs = res.GetNeighborIds();
  int32_t size = res.GetShape().size;
  std::cout << "TotalNeighborCount: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << nbrs[i] << std::endl;
  }
}

void TestFullSampleNeighbors(Client* client) {
  SamplingRequest req("click", "FullSampler", 3);
  int64_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  req.Set(ids, 10);

  SamplingResponse res;
  Status s = client->Sampling(&req, &res);
  std::cout << "FullSampleNeighbors: " << s.ToString() << std::endl;

  const int64_t* nbrs = res.GetNeighborIds();
  int32_t size = res.GetShape().size;
  std::cout << "TotalNeighborCount: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << nbrs[i] << std::endl;
  }

  auto& degrees = res.GetShape().segments;
  for (int32_t i = 0; i < degrees.size(); ++i) {
    std::cout << degrees[i] << std::endl;
  }
}

void TestNodeWeightNegativeSample(Client* client) {
  SamplingRequest req("user", "NodeWeightNegativeSampler", 3);
  int64_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  req.Set(ids, 10);

  SamplingResponse res;
  Status s = client->Sampling(&req, &res);
  std::cout << "NodeWeightNegativeSample: " << s.ToString() << std::endl;

  const int64_t* nbrs = res.GetNeighborIds();
  int32_t size = res.GetShape().size;
  std::cout << "TotalNeighborCount: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << nbrs[i] << std::endl;
  }
}

void TestNodeSubGraph(Client* client) {
  SubGraphRequest req("r");
  int64_t nodes[8];
  for (int32_t i = 0; i < 8; ++i) {
    nodes[i] = i;
  }
  req.Set(nodes, 8);
  SubGraphResponse res;
  Status s = client->SubGraph(&req, &res);
  std::cout << "TestNodeSubGraph: " << s.ToString() << std::endl;
  const int64_t* node_ids = res.NodeIds();
  const int32_t* rows = res.RowIndices();
  const int32_t* cols = res.ColIndices();
  const int64_t* edge_ids = res.EdgeIds();
  std::cout << "Node count: " << res.NodeCount() << std::endl;
  for (int32_t i = 0; i < res.NodeCount(); ++i) {
    std::cout << node_ids[i] << std::endl;
  }

  std::cout << "Edge count: " << res.EdgeCount() << std::endl;
  for (int32_t i = 0; i < res.EdgeCount(); ++i) {
    std::cout << rows[i] << "\t" << cols[i] << "\t" << edge_ids[i] << std::endl;
  }
}

void TestEdgeSubGraph(Client* client) {
  SubGraphRequest req("r");
  int64_t src[8];
  int64_t dst[8];
  for (int32_t i = 0; i < 8; ++i) {
    src[i] = i;
    dst[i] = i + 2;
  }
  req.Set(src, dst, 8);
  SubGraphResponse res;
  Status s = client->SubGraph(&req, &res);
  std::cout << "TestEdgeSubGraph: " << s.ToString() << std::endl;
  const int64_t* node_ids = res.NodeIds();
  const int32_t* rows = res.RowIndices();
  const int32_t* cols = res.ColIndices();
  const int64_t* edge_ids = res.EdgeIds();
  std::cout << "Node count: " << res.NodeCount() << std::endl;
  for (int32_t i = 0; i < res.NodeCount(); ++i) {
    std::cout << node_ids[i] << std::endl;
  }

  std::cout << "Edge count: " << res.EdgeCount() << std::endl;
  for (int32_t i = 0; i < res.EdgeCount(); ++i) {
    std::cout << rows[i] << "\t" << cols[i] << "\t" << edge_ids[i] << std::endl;
  }
}

void TestKHopSubGraph(Client* client) {
  int64_t nodes[8];
  for (int32_t i = 0; i < 8; ++i) {
    nodes[i] = i;
  }
  std::vector<int32_t> num_nbrs{2};
  SubGraphRequest req("r", num_nbrs);
  req.Set(nodes, 8);
  SubGraphResponse res;
  Status s = client->SubGraph(&req, &res);
  std::cout << "TestKHopSubGraph: " << s.ToString() << std::endl;
  const int64_t* node_ids = res.NodeIds();
  const int32_t* rows = res.RowIndices();
  const int32_t* cols = res.ColIndices();
  const int64_t* edge_ids = res.EdgeIds();
  std::cout << "Node count: " << res.NodeCount() << std::endl;
  for (int32_t i = 0; i < res.NodeCount(); ++i) {
    std::cout << node_ids[i] << std::endl;
  }

  std::cout << "Edge count: " << res.EdgeCount() << std::endl;
  for (int32_t i = 0; i < res.EdgeCount(); ++i) {
    std::cout << rows[i] << "\t" << cols[i] << "\t" << edge_ids[i] << std::endl;
  }
}

void TestGetCount(Client* client) {
  GetCountRequest req;
  GetCountResponse res;
  Status s = client->GetCount(&req, &res);
  std::cout << "GetCount: " << s.ToString() << std::endl;
  const int32_t* count = res.Count();
  int i = 0;
  for (; i < 4; ++i) {
    std::cout << "edge count: " << count[i] << std::endl;
  }
  for (int j = 0; j < 4; ++j) {
    std::cout << "node count: " << count[i+j] << std::endl;
  }
}

void TestGetStats(Client* client) {
  GetStatsRequest req;
  GetStatsResponse res;
  Status s = client->GetStats(&req, &res);
  std::cout << "GetStats: " << s.ToString() << std::endl;
  for(const auto& it : res.tensors_) {
    std::cout << "type " << it.first << ", count :" << std::endl;
    for (int32_t i = 0; i < it.second.Size(); ++i) {
      std::cout << it.second.GetInt32(i) << ",";
    }
    std::cout << std::endl;
  }
}

void TestConditionalNegativeSample(Client* client) {
  ConditionalSamplingRequest req("movie",
                                 "node_weight",
                                 5,
                                 "movie",
                                 true,
                                 true);
  int64_t src_ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int64_t dst_ids[10] = {0, 1, 12, 13, 14, 15, 16, 17, 18, 19};
  req.SetIds(src_ids, dst_ids, 10);
  std::vector<int32_t> int_cols={0};
  std::vector<float> int_props={0.5};
  std::vector<int32_t> float_cols;
  std::vector<float> float_props;
  std::vector<int32_t> str_cols={0};
  std::vector<float> str_props={0.5};
  req.SetSelectedCols(int_cols, int_props,
                      float_cols, float_props,
                      str_cols, str_props);
  SamplingResponse res;
  Status s = client->Sampling(&req, &res);
  std::cout << "ConditionalNegativeSample: " << s.ToString() << std::endl;

  const int64_t* nbrs = res.GetNeighborIds();
  int32_t size = res.GetShape().size;
  std::cout << "TotalNeighborCount: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << nbrs[i] << std::endl;
  }
}

int main(int argc, char** argv) {
  int32_t client_id = 0;
  int32_t client_count = 1;
  int32_t server_count = 1;
  std::string hosts;
  if (argc > 4) {
    client_id = argv[1][0] - '0';
    client_count = argv[2][0] - '0';
    server_count = argv[3][0] - '0';
    hosts = argv[4];
    SetGlobalFlagTrackerMode(kRpc);
    SetGlobalFlagServerHosts(hosts);
  } else if (argc > 3) {
    client_id = argv[1][0] - '0';
    client_count = argv[2][0] - '0';
    server_count = argv[3][0] - '0';
    SetGlobalFlagTrackerMode(kFileSystem);
    ::system("mkdir -p ./tracker");
    SetGlobalFlagTracker("./tracker");
  } else if (argc == 1) {
    ::system("mkdir -p ./tracker");
    SetGlobalFlagTracker("./tracker");
  } else {
    std::cout << "./client_test [client_id] [client_count] [server_count] [hosts]" << std::endl;
    std::cout << "e.g. ./client_test 0 2 2 127.0.0.1:8888,127.0.0.1:8889" << std::endl;
    std::cout << "or ./client_test [client_id] [client_count] [server_count]" << std::endl;
    std::cout << "e.g. ./client_test 0 2 2" << std::endl;
    return -1;
  }

  SetGlobalFlagDeployMode(kServer);
  SetGlobalFlagClientId(client_id);
  SetGlobalFlagClientCount(client_count);
  SetGlobalFlagServerCount(server_count);

  Client* client = NewRpcClient();

  TestGetEdges(client);
  TestGetNodes(client);
  TestLookupNodes(client);
  TestSumAggregateNodes(client);
  TestMeanAggregateNodes(client);
  TestMaxAggregateNodes(client);
  TestMinAggregateNodes(client);
  TestProdAggregateNodes(client);
  TestRandomSampleNeighbors(client);
  TestRandomWithoutReplacementSampleNeighbors(client);
  TestTopkSampleNeighbors(client);
  TestFullSampleNeighbors(client);
  TestNodeWeightNegativeSample(client);
  TestGetCount(client);
  TestConditionalNegativeSample(client);
  TestGetStats(client);
  TestNodeSubGraph(client);
  TestEdgeSubGraph(client);
  TestKHopSubGraph(client);

  Status s = client->Stop();
  std::cout << client_id << " in " << client_count << " client stop " << s.ToString() << std::endl;
  delete client;

  return 0;
}
