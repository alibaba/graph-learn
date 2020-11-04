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

#include "graphlearn/common/base/log.h"
#include "graphlearn/core/graph/storage/graph_storage.h"
#include "graphlearn/include/config.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class GraphStorageTest : public ::testing::Test {
public:
  GraphStorageTest() {
    InitGoogleLogging();
  }
  ~GraphStorageTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
    info_.type = "edge_type";
    info_.src_type = "src_type";
    info_.dst_type = "dst_type";
  }

  void TearDown() override {
  }

  void InternalTestOneNeighbor(GraphStorage* storage) {
    storage->Lock();
    storage->SetSideInfo(&info_);

    EdgeValue value;
    for (int32_t i = 0; i < 100; ++i) {
      value.attrs->Clear();
      GenEdgeValue(&value, i, 0);
      storage->Add(&value);
    }
    storage->Unlock();
    storage->Build();

    CheckInfo(storage->GetSideInfo());
    CheckEdges(storage, 0);
    CheckNeighbors(storage, 0);
  }

  void TestOneNeighbor() {
    GraphStorage* storage = NewMemoryGraphStorage();
    InternalTestOneNeighbor(storage);
    delete storage;
  }

  void TestOneNeighbor4CompressedStorage() {
    GraphStorage* storage = NewCompressedMemoryGraphStorage();
    InternalTestOneNeighbor(storage);
    delete storage;
  }

  void InternalTestTwoNeighbors(GraphStorage* storage) {
    storage->Lock();
    storage->SetSideInfo(&info_);

    EdgeValue value;
    for (int32_t i = 0; i < 100; ++i) {
      value.attrs->Clear();
      GenEdgeValue(&value, i, 0);
      storage->Add(&value);
    }
    for (int32_t i = 0; i < 100; ++i) {
      value.attrs->Clear();
      GenEdgeValue(&value, i, 100);
      storage->Add(&value);
    }
    storage->Unlock();
    storage->Build();

    CheckInfo(storage->GetSideInfo());
    CheckEdges(storage, 100);
    CheckNeighbors(storage, 100);
  }

  void TestTwoNeighbors() {
    GraphStorage* storage = NewMemoryGraphStorage();
    InternalTestTwoNeighbors(storage);
    delete storage;
  }

  void TestTwoNeighbors4CompressedStorage() {
    GraphStorage* storage = NewCompressedMemoryGraphStorage();
    InternalTestTwoNeighbors(storage);
    delete storage;
  }

  void GenEdgeValue(EdgeValue* value, int32_t index, int32_t dst_offset) {
    int32_t edge_index = index + dst_offset;

    value->src_id = index;
    value->dst_id = edge_index;
    if (info_.IsWeighted()) {
      value->weight = float(edge_index);
    }
    if (info_.IsLabeled()) {
      value->label = edge_index;
    }
    if (info_.IsAttributed()) {
      for (int32_t i = 0; i < info_.i_num; ++i) {
        value->attrs->Add(int64_t(edge_index + i));
      }
      for (int32_t i = 0; i < info_.f_num; ++i) {
        value->attrs->Add(float(edge_index + i));
      }
      for (int32_t i = 0; i < info_.s_num; ++i) {
        value->attrs->Add(std::to_string(edge_index + i));
      }
    }
  }

  void CheckInfo(const SideInfo* info) {
    EXPECT_EQ(info_.type, info->type);
    EXPECT_EQ(info_.src_type, info->src_type);
    EXPECT_EQ(info_.dst_type, info->dst_type);
    EXPECT_EQ(info_.format, info->format);
    EXPECT_EQ(info_.i_num, info->i_num);
    EXPECT_EQ(info_.f_num, info->f_num);
    EXPECT_EQ(info_.s_num, info->s_num);
  }

  void CheckEdges(GraphStorage* storage, IdType dst_offset) {
    IdType edge_count = storage->GetEdgeCount();
    EXPECT_EQ(edge_count, 100 + dst_offset);

    for (IdType edge_id = 0; edge_id < edge_count; ++edge_id) {
      IndexType label = storage->GetEdgeLabel(edge_id);
      if (info_.IsLabeled()) {
        EXPECT_EQ(label, IndexType(edge_id));
      } else {
        EXPECT_EQ(label, -1);
      }

      float weight = storage->GetEdgeWeight(edge_id);
      if (info_.IsWeighted()) {
        EXPECT_FLOAT_EQ(weight, float(edge_id));
      } else {
        EXPECT_FLOAT_EQ(weight, 0.0);
      }

      Attribute attr = storage->GetEdgeAttribute(edge_id);
      if (info_.IsAttributed()) {
        for (int32_t j = 0; j < info_.i_num; ++j) {
          EXPECT_EQ(attr->GetInts(nullptr)[j], IdType(edge_id + j));
        }
        for (int32_t j = 0; j < info_.f_num; ++j) {
          EXPECT_EQ(attr->GetFloats(nullptr)[j], float(edge_id + j));
        }
        for (int32_t j = 0; j < info_.s_num; ++j) {
          EXPECT_EQ(attr->GetStrings(nullptr)[j], std::to_string(edge_id + j));
        }
      } else {
        for (int32_t j = 0; j < info_.i_num; ++j) {
          EXPECT_EQ(attr->GetInts(nullptr)[j], GLOBAL_FLAG(DefaultIntAttribute));
        }
        for (int32_t j = 0; j < info_.f_num; ++j) {
          EXPECT_EQ(attr->GetFloats(nullptr)[j], GLOBAL_FLAG(DefaultFloatAttribute));
        }
        for (int32_t j = 0; j < info_.s_num; ++j) {
          EXPECT_EQ(attr->GetStrings(nullptr)[j], GLOBAL_FLAG(DefaultStringAttribute));
        }
      }
    }
  }

  void CheckNeighbors(GraphStorage* storage, IdType dst_offset) {
    IdType src_id_count = 100;
    IdType dst_id_count = 100 + dst_offset;

    const IdArray src_ids = storage->GetAllSrcIds();
    EXPECT_EQ(src_ids.Size(), src_id_count);
    for (IdType i = 0; i < src_id_count; ++i) {
      IdType src_id = src_ids.at(i);
      EXPECT_EQ(src_id, i);

      auto nbrs = storage->GetNeighbors(src_id);
      auto edges = storage->GetOutEdges(src_id);

      int32_t nbr_count = dst_offset / src_id_count + 1;
      EXPECT_EQ(nbrs.Size(), nbr_count);
      EXPECT_EQ(edges.Size(), nbr_count);
      EXPECT_EQ(storage->GetOutDegree(src_id), nbr_count);

      if (info_.IsWeighted()) {
        for (int32_t j = 0; j < nbr_count; ++j) {
          EXPECT_EQ(nbrs[j], src_id + (nbr_count - j - 1) * src_id_count);
          EXPECT_EQ(edges[j], src_id + (nbr_count - j - 1) * src_id_count);
        }
      } else {
        for (int32_t j = 0; j < nbr_count; ++j) {
          EXPECT_EQ(nbrs[j], src_id + j * src_id_count);
          EXPECT_EQ(edges[j], src_id + j * src_id_count);
        }
      }
    }

    const IndexList* out_degrees = storage->GetAllOutDegrees();
    EXPECT_EQ(out_degrees->size(), src_id_count);
    for (IdType i = 0; i < src_id_count; ++i) {
      EXPECT_EQ(out_degrees->at(i), 1 + dst_offset / src_id_count);
    }

    const IdArray dst_ids = storage->GetAllDstIds();
    EXPECT_EQ(dst_ids.Size(), dst_id_count);
    for (IdType i = 0; i < dst_id_count; ++i) {
      EXPECT_EQ(dst_ids.at(i), i);
      EXPECT_EQ(storage->GetInDegree(i), 1);
    }

    const IndexList* in_degrees = storage->GetAllInDegrees();
    EXPECT_EQ(in_degrees->size(), dst_id_count);
    for (IdType i = 0; i < dst_id_count; ++i) {
      EXPECT_EQ(in_degrees->at(i), 1);
    }
  }

protected:
  SideInfo info_;
};

TEST_F(GraphStorageTest, AddGetDefault) {
  info_.format = kDefault;
  TestOneNeighbor();
  TestOneNeighbor4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetWeighted) {
  info_.format = kWeighted;
  TestOneNeighbor();
  TestOneNeighbor4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetLabeled) {
  info_.format = kLabeled;
  TestOneNeighbor();
  TestOneNeighbor4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetAttributed) {
  info_.format = kAttributed;
  TestOneNeighbor();
  TestOneNeighbor4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetWeightedLabeled) {
  info_.format = kWeighted | kLabeled;
  TestOneNeighbor();
  TestOneNeighbor4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetWeightedAttributed) {
  info_.format = kWeighted | kAttributed;
  TestOneNeighbor();
  TestOneNeighbor4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetLabeledAttributed) {
  info_.format = kLabeled | kAttributed;
  TestOneNeighbor();
  TestOneNeighbor4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetWeightedLabeledAttributed) {
  info_.format = kWeighted | kLabeled | kAttributed;
  TestOneNeighbor();
  TestOneNeighbor4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetDefaultMultiNeighbors) {
  info_.format = kDefault;
  TestTwoNeighbors();
  TestTwoNeighbors4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetWeightedMultiNeighbors) {
  info_.format = kWeighted;
  TestTwoNeighbors();
  TestTwoNeighbors4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetLabeledMultiNeighbors) {
  info_.format = kLabeled;
  TestTwoNeighbors();
  TestTwoNeighbors4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetAttributedMultiNeighbors) {
  info_.format = kAttributed;
  TestTwoNeighbors();
  TestTwoNeighbors4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetWeightedLabeledMultiNeighbors) {
  info_.format = kWeighted | kLabeled;
  TestTwoNeighbors();
  TestTwoNeighbors4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetWeightedAttributedMultiNeighbors) {
  info_.format = kWeighted | kAttributed;
  TestTwoNeighbors();
  TestTwoNeighbors4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetLabeledAttributedMultiNeighbors) {
  info_.format = kLabeled | kAttributed;
  TestTwoNeighbors();
  TestTwoNeighbors4CompressedStorage();
}

TEST_F(GraphStorageTest, AddGetWeightedLabeledAttributedMultiNeighbors) {
  info_.format = kWeighted | kLabeled | kAttributed;
  TestTwoNeighbors();
  TestTwoNeighbors4CompressedStorage();
}
