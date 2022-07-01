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

#define GTEST_HAS_TR1_TUPLE 0
#include <iostream>
#include "graphlearn/include/status.h"
#include "graphlearn/common/base/errors.h"
#include "gtest/gtest.h"

using namespace graphlearn;  //NOLINT

TEST(Status, CopyAssign) {
  Status s1;
  EXPECT_TRUE(s1.ok());
  EXPECT_EQ(s1.ToString(), "OK");

  Status s2 = Status::OK();
  EXPECT_TRUE(s2.ok());
  EXPECT_EQ(s2.ToString(), "OK");

  Status s3(error::OUT_OF_RANGE, "out of range");
  s2 = s3;
  EXPECT_TRUE(!s2.ok());
  EXPECT_EQ(s2.code(), error::OUT_OF_RANGE);

  s3.Assign(error::OK);
  EXPECT_TRUE(s3.ok());
  EXPECT_EQ(s3.ToString(), "OK");

  Status s4(error::REQUEST_STOP, "request stop");
  s3 = s4;
  EXPECT_TRUE(!s3.ok());
  EXPECT_EQ(s3.code(), error::REQUEST_STOP);
}

TEST(Status, ErrorOp) {
  Status s1 = error::Cancelled("");
  EXPECT_TRUE(error::IsCancelled(s1));

  Status s2 = error::InvalidArgument("");
  EXPECT_TRUE(error::IsInvalidArgument(s2));

  Status s3 = error::NotFound("");
  EXPECT_TRUE(error::IsNotFound(s3));

  Status s4 = error::AlreadyExists("");
  EXPECT_TRUE(error::IsAlreadyExists(s4));

  Status s5 = error::ResourceExhausted("");
  EXPECT_TRUE(error::IsResourceExhausted(s5));

  Status s6 = error::Unavailable("");
  EXPECT_TRUE(error::IsUnavailable(s6));

  Status s7 = error::FailedPrecondition("");
  EXPECT_TRUE(error::IsFailedPrecondition(s7));

  Status s8 = error::OutOfRange("");
  EXPECT_TRUE(error::IsOutOfRange(s8));

  Status s9 = error::Unimplemented("");
  EXPECT_TRUE(error::IsUnimplemented(s9));

  Status s10 = error::Internal("");
  EXPECT_TRUE(error::IsInternal(s10));

  Status s11 = error::Aborted("");
  EXPECT_TRUE(error::IsAborted(s11));

  Status s12 = error::DeadlineExceeded("");
  EXPECT_TRUE(error::IsDeadlineExceeded(s12));

  Status s13 = error::DataLoss("");
  EXPECT_TRUE(error::IsDataLoss(s13));

  Status s14 = error::Unknown("");
  EXPECT_TRUE(error::IsUnknown(s14));

  Status s15 = error::PermissionDenied("");
  EXPECT_TRUE(error::IsPermissionDenied(s15));

  Status s16 = error::Unauthenticated("");
  EXPECT_TRUE(error::IsUnauthenticated(s16));

  Status s17 = error::RequestStop("");
  EXPECT_TRUE(error::IsRequestStop(s17));
}

