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

#include <sys/time.h>

namespace graphlearn {

// ::google::IsGoogleLoggingInitialized() requires a relative new version
// of glog, thus we use our own static initialization guard.
static bool _is_google_logging_initialized = false;

void InitGoogleLogging() {
  if (_is_google_logging_initialized) {
    return;
  }
  _is_google_logging_initialized = true;

  FLAGS_alsologtostderr = false;
  FLAGS_colorlogtostderr = true;
  FLAGS_log_dir = ".";
  FLAGS_minloglevel = ::google::INFO;
  ::google::InitGoogleLogging("graphlearn");
}

void UninitGoogleLogging() {
  if (!_is_google_logging_initialized) {
    return;
  }
  _is_google_logging_initialized = false;

  ::google::ShutdownGoogleLogging();
}


void Log(const char* msg) {
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  struct tm rslt;
  struct tm* p = gmtime_r(&tv.tv_sec, &rslt);

  fprintf(stderr,
          "[%04d-%02d-%02d %02d:%02d:%02d.%ld] %s\n",
          1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday,
          8 + p->tm_hour, p->tm_min, p->tm_sec, tv.tv_usec,
          msg);
}

void Log(const std::string& msg) {
  Log(msg.c_str());
}

}  // namespace graphlearn
