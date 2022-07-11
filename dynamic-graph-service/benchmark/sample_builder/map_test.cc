/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#include <cstdio>
#include <string>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <vector>

struct Prefix {
  uint64_t   vtype;
  uint64_t   vid;
  uint64_t   op_id;

  Prefix(uint64_t vtype, uint64_t vid, uint64_t op_id)
      : vtype(vtype), vid(vid), op_id(op_id) {}

  inline bool operator==(const Prefix &p) const {
    return vtype == p.vtype && vid == p.vid && op_id == p.op_id;
  }
};

struct Hasher {
  std::size_t operator() (const Prefix &p) const {
    std::size_t h1 = std::hash<uint64_t>()(p.vtype);
    std::size_t h2 = std::hash<uint64_t>()(p.vid);
    std::size_t h3 = std::hash<uint64_t>()(p.op_id);
    return h1 ^ h2 ^ h3;
  }
};

template<typename T>
void WriteReadTest(uint64_t N) {
  T map;
  // Insert N elements in order
  auto start = std::chrono::steady_clock::now();
  for(uint64_t id = 0; id < N; id++) {
    map.emplace(Prefix(0, id, 0), id);
  }
  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
  std::cout<< "Write throughput : " << (float)N / elapsed.count() << " ops/sec. \n";

  // Ramdomly read N elements, repeat for 10 times
  int ret = 0;
  for (int repeat = 1; repeat <= 10; repeat++) {
    std::vector<Prefix> idx;
    for(uint64_t id = 0; id < N; id++) {
      Prefix pkey(0, std::rand() % N, 0);
      idx.push_back(pkey);
    }

    start = std::chrono::steady_clock::now();
    for(uint64_t id = 0; id < N; id++) {
      ret += map[idx[id]];
    }
    elapsed += std::chrono::steady_clock::now() - start;
  }
  std::cout<< "Random read throughput : "<< (float)(N * 10) / elapsed.count() << " ops/sec \n";
}

int main(int argc, char** argv) {
  int N = std::atoi(argv[1]);
  WriteReadTest<std::unordered_map<Prefix, uint64_t, Hasher>>(N);
}
