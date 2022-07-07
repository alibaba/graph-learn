/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "gtest/gtest.h"

#include "service/request/query_request.h"

using namespace dgs;
using namespace seastar;

TEST(UnInstallQueryRequestMain, Construct) {
  std::string uninstall_query_req_json = "{ \
      query_id: 1 \
    }";

  std::string schemafile;
  std::string jsonfile;
  bool ok = flatbuffers::LoadFile(
    "../../fbs/uninstall_query_req.fbs", false, &schemafile);
  EXPECT_EQ(ok, true);

  flatbuffers::Parser parser;
  parser.Parse(schemafile.c_str());
  parser.Parse(uninstall_query_req_json.c_str());

  uint8_t* buf = parser.builder_.GetBufferPointer();
  auto size = parser.builder_.GetSize();

  temporary_buffer<char> tp(reinterpret_cast<char*>(buf), size);
  UnInstallQueryRequest req(std::move(tp));

  EXPECT_EQ(req.query_id(), 1);
}
