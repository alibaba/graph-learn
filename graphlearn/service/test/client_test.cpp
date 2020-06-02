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
#include "graphlearn/include/client.h"
#include "graphlearn/include/config.h"

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
      const std::string* strs = res.StringAttrs();
      std::cout << "strings: ";
      for (int32_t i = 0; i < size * str_num; ++i) {
        std::cout << strs[i] << ' ';
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
  std::cout << "SampleNeighbors: " << s.ToString() << std::endl;

  const int64_t* nbrs = res.GetNeighborIds();
  int32_t size = res.TotalNeighborCount();
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
  int32_t size = res.TotalNeighborCount();
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
  int32_t size = res.TotalNeighborCount();
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
  int32_t size = res.TotalNeighborCount();
  std::cout << "TotalNeighborCount: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << nbrs[i] << std::endl;
  }

  int32_t batch_size = res.BatchSize();
  const int32_t* degrees = res.GetDegrees();
  for (int32_t i = 0; i < batch_size; ++i) {
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
  int32_t size = res.TotalNeighborCount();
  std::cout << "TotalNeighborCount: " << size << std::endl;
  for (int32_t i = 0; i < size; ++i) {
    std::cout << nbrs[i] << std::endl;
  }
}


int main(int argc, char** argv) {
  ::system("mkdir -p ./tracker");

  int32_t client_id = 0;
  int32_t client_count = 1;
  int32_t server_count = 1;
  if (argc == 1) {
  } else if (argc > 3) {
    client_id = argv[1][0] - '0';
    client_count = argv[2][0] - '0';
    server_count = argv[3][0] - '0';
  } else {
    std::cout << "./client_test [client_id] [client_count] [server_count]" << std::endl;
    std::cout << "e.g. ./client_test 0 2 2" << std::endl;
    return -1;
  }

  SetGlobalFlagDeployMode(1);
  SetGlobalFlagClientId(client_id);
  SetGlobalFlagClientCount(client_count);
  SetGlobalFlagServerCount(server_count);
  SetGlobalFlagTracker("./tracker");

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

  Status s = client->Stop();
  std::cout << client_id << " in " << client_count << " client stop " << s.ToString() << std::endl;
  delete client;

  return 0;
}
