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

#include "graphlearn/common/rpc/notification.h"

#include <iostream>
#include "gtest/gtest.h"
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"

using namespace graphlearn;  //NOLINT
using namespace ::testing;  //NOLINT

class RpcNotificationTest : public ::testing::Test {
protected:
  void SetUp() override {
    InitGoogleLogging();
    notification_ = new RpcNotification;
  }

  void TearDown() override {
    UninitGoogleLogging();
    delete notification_;
  }

protected:
  RpcNotification* notification_;
};

TEST_F(RpcNotificationTest, Normal) {
  notification_->Init("normal", 2);
  notification_->SetCallback([](const std::string& req_type,
                                const Status& status){
        EXPECT_TRUE(status.ok());
        std::cout << "RpcNotification ok." << std::endl;
      });
  EXPECT_EQ(notification_->AddRpcTask(0), 1);
  EXPECT_EQ(notification_->AddRpcTask(1), 2);
  notification_->Notify(1);
  notification_->Notify(0);
}

TEST_F(RpcNotificationTest, Timeout) {
  notification_->Init("normal", 2);
  notification_->SetCallback([](const std::string& req_type,
                                const Status& status){
        EXPECT_TRUE(!status.ok());
        std::cout << "RpcNotification failed: "
                  << status.ToString()
                  << std::endl;
      });
  EXPECT_EQ(notification_->AddRpcTask(0), 1);
  EXPECT_EQ(notification_->AddRpcTask(3), 2);
  notification_->Notify(0);
  notification_->Wait(1);
}

TEST_F(RpcNotificationTest, Fail) {
  notification_->Init("fail", 2);
  notification_->SetCallback([](const std::string& req_type,
                                const Status& status){
        EXPECT_TRUE(!status.ok());
        std::cout << "RpcNotification failed: "
                  << status.ToString()
                  << std::endl;
      });
  EXPECT_EQ(notification_->AddRpcTask(0), 1);
  EXPECT_EQ(notification_->AddRpcTask(3), 2);
  notification_->Notify(0);
  notification_->Notify(1);     // not exists rpc id
  notification_->Notify(0);     // repeat rpc id
  notification_->NotifyFail(3, error::Internal("failure."));  // notified
}
