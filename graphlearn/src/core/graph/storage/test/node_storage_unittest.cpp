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

#include "common/base/log.h"
#include "core/graph/storage/node_storage.h"
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]
using namespace graphlearn::io;  // NOLINT [build/namespaces]

class NodeStorageTest : public ::testing::Test {
public:
  NodeStorageTest() {
    InitGoogleLogging();
  }
  ~NodeStorageTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
    info_.type = "node_type";
  }

  void TearDown() override {
  }

  void InternalTest(NodeStorage* storage) {
    storage->SetSideInfo(&info_);

    storage->Lock();
    NodeValue value;
    for (int32_t i = 0; i < 100; ++i) {
      value.attrs->Clear();
      GenNodeValue(&value, i);
      storage->Add(&value);
    }
    storage->Unlock();
    storage->Build();

    CheckInfo(storage->GetSideInfo());
    CheckNodes(storage);
  }

  void Test() {
    NodeStorage* storage = NewMemoryNodeStorage();
    InternalTest(storage);
    const std::vector<Attribute>* attrs = storage->GetAttributes();
    if (!info_.IsAttributed()) {
      EXPECT_EQ(attrs->size(), 0);
    }
    delete storage;
  }

  void Test4CompressedStorage() {
    NodeStorage* storage = NewCompressedMemoryNodeStorage();
    InternalTest(storage);
    const std::vector<Attribute>* attrs = storage->GetAttributes();
    EXPECT_EQ(attrs, nullptr);
    delete storage;
  }

  void GenNodeValue(NodeValue* value, int32_t node_index) {
    value->id = node_index;
    if (info_.IsWeighted()) {
      value->weight = float(node_index);
    }
    if (info_.IsLabeled()) {
      value->label = node_index;
    }
    if (info_.IsAttributed()) {
      for (int32_t i = 0; i < info_.i_num; ++i) {
        value->attrs->Add(int64_t(node_index + i));
      }
      for (int32_t i = 0; i < info_.f_num; ++i) {
        value->attrs->Add(float(node_index + i));
      }
      for (int32_t i = 0; i < info_.s_num; ++i) {
        value->attrs->Add(std::to_string(node_index + i));
      }
    }
  }

  void CheckInfo(const SideInfo* info) {
    EXPECT_EQ(info_.type, info->type);
    EXPECT_EQ(info_.format, info->format);
    EXPECT_EQ(info_.i_num, info->i_num);
    EXPECT_EQ(info_.f_num, info->f_num);
    EXPECT_EQ(info_.s_num, info->s_num);
  }

  void CheckNodes(NodeStorage* storage) {
    IdType node_count = storage->Size();
    EXPECT_EQ(node_count, 100);

    for (IdType node_id = 0; node_id < node_count; ++node_id) {
      IndexType label = storage->GetLabel(node_id);
      if (info_.IsLabeled()) {
        EXPECT_EQ(label, IndexType(node_id));
      } else {
        EXPECT_EQ(label, -1);
      }

      float weight = storage->GetWeight(node_id);
      if (info_.IsWeighted()) {
        EXPECT_FLOAT_EQ(weight, float(node_id));
      } else {
        EXPECT_FLOAT_EQ(weight, 0.0);
      }

      Attribute attr = storage->GetAttribute(node_id);
      if (info_.IsAttributed()) {
        for (int32_t j = 0; j < info_.i_num; ++j) {
          EXPECT_EQ(attr->GetInts(nullptr)[j], IdType(node_id + j));
        }
        for (int32_t j = 0; j < info_.f_num; ++j) {
          EXPECT_EQ(attr->GetFloats(nullptr)[j], float(node_id + j));
        }
        for (int32_t j = 0; j < info_.s_num; ++j) {
          EXPECT_EQ(attr->GetStrings(nullptr)[j], std::to_string(node_id + j));
        }
      } else {
        EXPECT_TRUE(attr.get() == nullptr);
      }
    }

    const IdList* node_ids = storage->GetIds();
    EXPECT_EQ(node_ids->size(), node_count);
    for (IdType index = 0; index < node_count; ++index) {
      EXPECT_EQ(node_ids->at(index), index);
    }

    const IndexList* labels = storage->GetLabels();
    if (info_.IsLabeled()) {
      EXPECT_EQ(labels->size(), node_count);
      for (IdType index = 0; index < node_count; ++index) {
        EXPECT_EQ(labels->at(index), index);
      }
    } else {
      EXPECT_EQ(labels->size(), 0);
    }

    const std::vector<float>* weights = storage->GetWeights();
    if (info_.IsWeighted()) {
      EXPECT_EQ(weights->size(), node_count);
      for (IdType index = 0; index < node_count; ++index) {
        EXPECT_EQ(weights->at(index), float(index));
      }
    } else {
      EXPECT_EQ(weights->size(), 0);
    }
  }

protected:
  SideInfo info_;
};

TEST_F(NodeStorageTest, AddGetWeighted) {
  info_.format = kWeighted;
  Test();
  Test4CompressedStorage();
}

TEST_F(NodeStorageTest, AddGetLabeled) {
  info_.format = kLabeled;
  Test();
  Test4CompressedStorage();
}

TEST_F(NodeStorageTest, AddGetAttributed) {
  info_.format = kAttributed;
  Test();
  Test4CompressedStorage();
}

TEST_F(NodeStorageTest, AddGetWeightedLabeled) {
  info_.format = kWeighted | kLabeled;
  Test();
  Test4CompressedStorage();
}

TEST_F(NodeStorageTest, AddGetWeightedAttributed) {
  info_.format = kWeighted | kAttributed;
  Test();
  Test4CompressedStorage();
}

TEST_F(NodeStorageTest, AddGetLabeledAttributed) {
  info_.format = kLabeled | kAttributed;
  Test();
  Test4CompressedStorage();
}

TEST_F(NodeStorageTest, AddGetWeightedLabeledAttributed) {
  info_.format = kWeighted | kLabeled | kAttributed;
  Test();
  Test4CompressedStorage();
}
