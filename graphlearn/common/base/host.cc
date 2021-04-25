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

#include "graphlearn/common/base/host.h"

#include <arpa/inet.h>
#include <unistd.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <vector>
#include "graphlearn/common/base/log.h"

namespace graphlearn {

std::string GetLocalEndpoint(int32_t port) {
  char host_name[128] = {0};
  int ret = gethostname(host_name, sizeof(host_name));
  if (ret < 0) {
    LOG(FATAL) << "gethostname error: " << ret;
    return "";
  }

  hostent* hptr = gethostbyname(host_name);
  if (hptr == NULL) {
    LOG(FATAL) << "gethostbyname error";
    return "";
  }

  int i = 0;
  while (hptr->h_addr_list[i] != NULL) {
    std::string ip = inet_ntoa(*(struct in_addr*)hptr->h_addr_list[i]);
    if (ip != "127.0.0.1") {
      return ip + ":" + std::to_string(port);
    } else {
      ++i;
    }
  }
  return "";
}

int32_t GetAvailablePort() {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    LOG(FATAL) << "GetAvailablePort with socket error.";
    return -1;
  }
  struct sockaddr_in serv_addr;
  bzero(reinterpret_cast<char*>(&serv_addr), sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  // auto-detect port.
  serv_addr.sin_port = 0;
  if (bind(sock, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
    LOG(FATAL) << "GetAvailablePort failed with auto-binding port.";
    return -1;
  }

  socklen_t len = sizeof(serv_addr);
  if (getsockname(sock, (struct sockaddr *)&serv_addr, &len) == -1) {
    LOG(FATAL) << "GetAvailablePort failed with geting socket name.";
    return -1;
  }
  if (close(sock) < 0) {
    LOG(FATAL) << "GetAvailablePort failed with closing socket.";
    return -1;
  }
  return int32_t(ntohs(serv_addr.sin_port));
}

}  // namespace graphlearn
