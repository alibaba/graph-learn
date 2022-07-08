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

#include "service/test/test_helper.h"

using namespace dgs;

class ServingActorModuleTest : public ::testing::Test {
public:
  ServingActorModuleTest() : helper_() {}
  ~ServingActorModuleTest() override = default;

protected:
  void SetUp() override {
    helper_.Initialize();
  }

  void TearDown() override {
    helper_.Finalize();
  }

  void MakeSampleStore() {
    helper_.MakeSampleStore();
  }

  InstallQueryRequest MakeInstallQueryRequest() {
    return helper_.MakeInstallQueryRequest();
  }

  RunQueryRequest MakeRunQueryRequest(VertexId vid) {
    return helper_.MakeRunQueryRequest(vid);
  }

  void PrintQueryResponse(const QueryResponse& res) {
    helper_.PrintQueryResponse(res);
  }

protected:
  TestHelper  helper_;
  ActorSystem actor_system_{WorkerType::Serving, 0, 1, 2};
};

TEST_F(ServingActorModuleTest, RunQuery) {
  MakeSampleStore();

  // install query and run.
  auto fut = seastar::alien::submit_to(
      *seastar::alien::internal::default_instance, 0, [this] {
    // create actor reference to shard 0.
    auto builder = hiactor::scope_builder(0, MakeServingGroupScope());
    auto ref = MakeServingActorInstRef(builder);

    auto req = MakeInstallQueryRequest();
    auto payload = std::make_shared<ServingInitPayload>(
        req.CloneBuffer(), helper_.GetSampleStore());

    return seastar::do_with(std::move(ref), [this, payload] (ServingActor_ref& ref) {
      return ref.ExecuteAdminOperation(AdminRequest(AdminOperation::INIT, payload)).then([ref, this] (auto) mutable {
        return ref.RunQuery(this->MakeRunQueryRequest(2));
      });
    });
  });
  fut.wait();

  PrintQueryResponse(fut.get());
}
