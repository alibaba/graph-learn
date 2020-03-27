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
#include <vector>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/include/constants.h"
#include "graphlearn/platform/env.h"
#include "gtest/gtest.h"

using namespace graphlearn;  //NOLINT [build/namespaces]

class LocalFSTest : public ::testing::Test {
public:
  LocalFSTest() {
    InitGoogleLogging();
  }
  ~LocalFSTest() {
    UninitGoogleLogging();
  }

protected:
  void SetUp() override {
    env_ = Env::Default();
  }

  void TearDown() override {
  }

  void GenBytesFile(const char* file_name) {
    FileSystem* fs = NULL;
    Status s = env_->GetFileSystem(file_name, &fs);
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(fs != NULL);

    std::unique_ptr<WritableFile> f;
    s = fs->NewWritableFile(file_name, &f);
    EXPECT_TRUE(s.ok());

    char alpha[26];
    for (int32_t i = 0; i < 26; ++i) {
      alpha[i] = 'A' + i;
    }

    s = f->Append(LiteString(alpha, 26));
    EXPECT_TRUE(s.ok());
    s = f->Flush();
    EXPECT_TRUE(s.ok());
    s = f->Close();
    EXPECT_TRUE(s.ok());
  }

  void GenStructuredFile(const char* file_name) {
    FileSystem* fs = NULL;
    Status s = env_->GetFileSystem(file_name, &fs);
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(fs != NULL);

    std::unique_ptr<WritableFile> f;
    s = fs->NewWritableFile(file_name, &f);
    EXPECT_TRUE(s.ok());

    const char* title = "f1:int32\tf2:int64\tf3:float\tf4:string\n";
    s = f->Append(LiteString(title, strlen(title)));
    EXPECT_TRUE(s.ok());

    char buffer[32];
    for (int32_t i = 0; i < 26; ++i) {
      char c = char('A' + i);
      int32_t size = snprintf(buffer, sizeof(buffer),
                              "%d\t%d\t%f\t\%c%c\n", i, i, float(i), c, c);
      s = f->Append(LiteString(buffer, size));
      EXPECT_TRUE(s.ok());
    }

    s = f->Flush();
    EXPECT_TRUE(s.ok());
    s = f->Close();
    EXPECT_TRUE(s.ok());
  }

  void TestByteStreamRead(const char* file_name, int64_t offset) {
    FileSystem* fs = NULL;
    Status s = env_->GetFileSystem(file_name, &fs);
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(fs != NULL);

    std::unique_ptr<ByteStreamAccessFile> reader;
    s = fs->NewByteStreamAccessFile(file_name, offset, &reader);
    EXPECT_TRUE(s.ok());

    LiteString ret;
    char buffer[26];
    s = reader->Read(26, &ret, buffer + offset);
    EXPECT_TRUE(s.ok());
    for (int32_t i = offset; i < 26; ++i) {
      EXPECT_TRUE(buffer[i] == 'A' + i);
    }

    s = reader->Read(1, &ret, buffer);
    EXPECT_TRUE(error::IsOutOfRange(s));

    s = fs->FileExists(file_name);
    EXPECT_TRUE(s.ok());

    uint64_t size = 0;
    s = fs->GetFileSize(file_name, &size);
    EXPECT_TRUE(size == 26);

    std::string name = fs->Translate(file_name);
    EXPECT_TRUE(name == std::string(file_name));

    s = fs->DeleteFile(file_name);
    EXPECT_TRUE(s.ok());

    s = fs->FileExists(file_name);
    EXPECT_TRUE(error::IsNotFound(s));
  }

  void TestStructuredRead(const char* file_name, int64_t offset) {
    FileSystem* fs = NULL;
    Status s = env_->GetFileSystem(file_name, &fs);
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(fs != NULL);

    std::unique_ptr<StructuredAccessFile> reader;
    s = fs->NewStructuredAccessFile(file_name, offset, &reader);
    EXPECT_TRUE(s.ok());

    const io::Schema& schema = reader->GetSchema();
    EXPECT_EQ(schema.Size(), 4);
    EXPECT_EQ(schema.types[0], DataType::kInt32);
    EXPECT_EQ(schema.types[1], DataType::kInt64);
    EXPECT_EQ(schema.types[2], DataType::kFloat);
    EXPECT_EQ(schema.types[3], DataType::kString);

    io::Record ret;
    ret.Reserve(schema.Size());
    for (int32_t i = offset; i < 26; ++i) {
      s = reader->Read(&ret);
      EXPECT_TRUE(s.ok());
      EXPECT_EQ(ret[0].n.i, i);
      EXPECT_EQ(ret[1].n.l, int64_t(i));
      EXPECT_EQ(ret[2].n.f, float(i));
      EXPECT_EQ(ret[3].s.len, 2);

      std::string s;
      s.push_back(char('A' + i));
      s.push_back(char('A' + i));
      EXPECT_EQ(std::string(ret[3].s.data, ret[3].s.len), s);
    }

    s = reader->Read(&ret);
    EXPECT_TRUE(error::IsOutOfRange(s));

    s = fs->FileExists(file_name);
    EXPECT_TRUE(s.ok());

    uint64_t size = 0;
    s = fs->GetRecordCount(file_name, &size);
    EXPECT_TRUE(size == 26);

    std::string name = fs->Translate(file_name);
    EXPECT_TRUE(name == std::string(file_name));

    s = fs->DeleteFile(file_name);
    EXPECT_TRUE(s.ok());

    s = fs->FileExists(file_name);
    EXPECT_TRUE(error::IsNotFound(s));
  }

protected:
  Env* env_;
};

TEST_F(LocalFSTest, Directory) {
  FileSystem* fs = NULL;
  Status s = env_->GetFileSystem("test_dir", &fs);
  EXPECT_TRUE(s.ok());
  EXPECT_TRUE(fs != NULL);

  s = fs->CreateDir("test_dir");
  EXPECT_TRUE(s.ok());
  s = fs->CreateDir("test_dir/00");
  EXPECT_TRUE(s.ok());

  GenBytesFile("test_dir/11");
  GenBytesFile("test_dir/22");

  // list dir
  std::vector<std::string> files;
  s = fs->ListDir("test_dir", &files);
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(files.size(), 3);

  for (size_t i = 0; i < files.size(); ++i) {
    if (files[i] == "11") {
      EXPECT_TRUE(true);
    } else if (files[i] == "22") {
      EXPECT_TRUE(true);
    } else if (files[i] == "00/") {
      EXPECT_TRUE(true);
    } else {
      EXPECT_TRUE(false);
    }
  }

  // list file
  files.clear();
  s = fs->ListDir("test_dir/11", &files);
  EXPECT_TRUE(!s.ok());

  s = fs->DeleteDir("test_dir/");
  EXPECT_TRUE(!s.ok());

  s = fs->DeleteFile("test_dir/11");
  EXPECT_TRUE(s.ok());

  s = fs->DeleteFile("test_dir/22");
  EXPECT_TRUE(s.ok());

  s = fs->DeleteDir("test_dir/00/");
  EXPECT_TRUE(s.ok());

  s = fs->DeleteDir("test_dir/");
  EXPECT_TRUE(s.ok());
}

TEST_F(LocalFSTest, ByteStreamReadNoOffset) {
  const char* file_name = "test_local_file";
  GenBytesFile(file_name);
  TestByteStreamRead(file_name, 0);
}

TEST_F(LocalFSTest, ByteStreamReadOffset) {
  const char* file_name = "test_local_file";
  GenBytesFile(file_name);
  TestByteStreamRead(file_name, 10);
}

TEST_F(LocalFSTest, StructuredReadNoOffset) {
  const char* file_name = "test_structured_local_file";
  GenStructuredFile(file_name);
  TestStructuredRead(file_name, 0);
}

TEST_F(LocalFSTest, StructuredReadOffset) {
  const char* file_name = "test_structured_local_file";
  GenStructuredFile(file_name);
  TestStructuredRead(file_name, 16);
}
