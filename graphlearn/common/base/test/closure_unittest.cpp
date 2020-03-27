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

#include "graphlearn/common/base/closure.h"

#include <string>
#include "gtest/gtest.h"

using namespace graphlearn;  // NOLINT [build/namespaces]

namespace {

int foobar(char c) {
  return 1;
}

struct Foo {
  int Bar(int i, const std::string*s, double d) {
    return -1;
  }
};

}  // anonymous namespace


TEST(ClosureTest, FunctionWithoutPrebind) {
  Closure<int, char>* c = NewClosure(foobar);
  EXPECT_EQ(c->Run('a'), 1);
}

TEST(ClosureTest, FunctionWithPrebind) {
  Closure<int>* c = NewClosure(foobar, 'a');
  EXPECT_EQ(c->Run(), 1);
}

TEST(ClosureTest, PermanentFunctionWithOutPrebind) {
  Closure<int, char>* c = NewPermanentClosure(foobar);
  EXPECT_EQ(c->Run('a'), 1);
  EXPECT_EQ(c->Run('a'), 1);
  delete c;
}

TEST(ClosureTest, PermanentFunctionWithPrebind) {
  Closure<int>* c = NewPermanentClosure(foobar, 'a');
  EXPECT_EQ(c->Run(), 1);
  EXPECT_EQ(c->Run(), 1);
  delete c;
}

TEST(ClosureTest, MethodWithoutPrebind) {
  Foo foo;
  Closure<int, int, const std::string*, double>* c =
    NewClosure(&foo, &Foo::Bar);
  EXPECT_EQ(c->Run(0, nullptr, 1.0), -1);
}

TEST(ClosureTest, MethodWithPrebind) {
  Foo foo;
  Closure<int, double>* c =
    NewClosure(&foo, &Foo::Bar, 1, static_cast<const std::string*>(nullptr));
  EXPECT_EQ(c->Run(1.0), -1);
}

TEST(ClosureTest, PermanentMethodWithoutPrebind) {
  Foo foo;
  Closure<int, int, const std::string*, double>* c =
    NewPermanentClosure(&foo, &Foo::Bar);
  EXPECT_EQ(c->Run(0, nullptr, 1.0), -1);
  EXPECT_EQ(c->Run(0, nullptr, 1.0), -1);
  delete c;
}

TEST(ClosureTest, PermanentMethodWithPrebind) {
  Foo foo;
  Closure<int, double>* c =
    NewPermanentClosure(&foo, &Foo::Bar, 1,
        static_cast<const std::string*>(nullptr));
  EXPECT_EQ(c->Run(1.0), -1);
  EXPECT_EQ(c->Run(1.0), -1);
  delete c;
}

