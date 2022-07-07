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

#include "common/base/base64.h"

namespace graphlearn {

namespace {

static const unsigned char EncodeTable[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static const unsigned char* DecodeTable = NULL;

void FillDecodeTable() {
  // double buffer is used here for thread safe.
  static unsigned char DecodeTableBuff[256];
  unsigned char buff[256];
  if (DecodeTable != NULL) {
    return;
  }
  ::memset(buff, 0x80, sizeof(buff));

  for (size_t k = 0; k < sizeof(EncodeTable); ++k) {
    buff[(size_t)EncodeTable[k]] = k;
  }
  // to mark those valid characters in encoded string, but not in these
  // 64 bases list.
  buff[(size_t)'\r'] = buff[(size_t)'\n'] = 0x4F;
  buff[(size_t)'='] = 0x40;

  ::memcpy(DecodeTableBuff, buff, sizeof(DecodeTableBuff));
  DecodeTable = DecodeTableBuff;
}

// Get the next 4 characters from input string, '\r\n' will be trimmed off.
// The input string starts from 'p', and ends before 'q'. 'buff' is for
// storing the return characters.
// The return value, -1: error, there aren't 4 characters available, or get
// invalid character. 0-4 mean the number of valid characters, '=' is excluded.
int GetNext4EncodedCharacters(const unsigned char*& p,  // NOLINT
                              const unsigned char* q,
                              unsigned char* buff) {
  int k = 0;
  unsigned char c = 0;
  while (k < 4 && p < q) {
    c = DecodeTable[*p];
    if ((c & 0xC0) == 0) {      // normal valid characters
      *buff++ = c;
      ++p;
      ++k;
    } else if (c & 0x80) {      // not ( '\r' or '\n' or '=' )
      return -1;
    } else if (*p == '=') {
      break;
    } else {                    // ( '\r' or '\n' )
      ++p;
    }
  }
  // success. this should be most of the cases, return as soon as possible
  if (k == 4) {
    return 4;
  }
  // get a '='
  if (p < q && *p == '=') {
    ++p;
    // there should be 4 characters in the last encode group
    int tail = 4 - k - 1;
    // there should not be more than 2 '=' in the end
    if (tail > 1) {
      return -1;
    }
    while (tail > 0 && p < q && ((DecodeTable[*p] & 0x40) == 0x40)) {
      if (*p == '=') {
        --tail;
      }
      ++p;
    }
    // any character not ('\r' or '\n' or '=') appears after '='
    if (tail != 0) {
      return -1;
    }
    // only ('\r' || '\n') is allowed at the end
    while (p < q) {
      if ((DecodeTable[*p] & 0x4F) == 0x4F) {
        ++p;
      } else {
        return -1;
      }
    }

    return k;
  }
  // for ( '\r' or '\n' ) at very end
  while (p < q && (DecodeTable[*p] & 0x4F) == 0x4F) {
    ++p;
  }
  if (k == 0 && p == q) {
    return 0;
  }

  return -1;
}

size_t ExpectedEncodeLength(size_t len) {
  size_t encodedLen = ((len * 4 / 3 + 3) / 4) * 4;
  return encodedLen;
}

size_t ExpectedDecodeLength(size_t len) {
  return (size_t)((len + 3) / 4 * 3);
}

}  // anonymous namespace

bool Base64Encode(const LiteString& input, std::string* output) {
  output->resize(ExpectedEncodeLength(input.size()));

  char* buff = &(*output)[0];
  size_t len = output->size();

  if (!Base64Encode(input, buff, &len)) {
    output->clear();
    return false;
  }
  output->resize(len);
  return true;
}

bool Base64Encode(const LiteString& input, char* output, size_t* len) {
  char* buff = output;
  if (__builtin_expect(*len < ExpectedEncodeLength(input.size()), 0)) {
    return false;
  }

  unsigned char *p = (unsigned char*)input.data();
  unsigned char *q = p + input.size();
  unsigned char c1, c2, c3;

  // process 3 char every loop
  for ( ; p + 3 <= q; p += 3) {
    c1 = *p;
    c2 = *(p + 1);
    c3 = *(p + 2);

    *buff++ = EncodeTable[c1 >> 2];
    *buff++ = EncodeTable[((c1 << 4) | (c2 >> 4)) & 0x3f];
    *buff++ = EncodeTable[((c2 << 2) | (c3 >> 6)) & 0x3f];
    *buff++ = EncodeTable[c3 & 0x3f];
  }

  // the reminders
  if (q - p == 1) {
    c1 = *p;
    *buff++ = EncodeTable[(c1 & 0xfc) >> 2];
    *buff++ = EncodeTable[(c1 & 0x03) << 4];
    *buff++ = '=';
    *buff++ = '=';
  } else if (q - p == 2) {
    c1 = *p;
    c2 = *(p + 1);
    *buff++ = EncodeTable[(c1 & 0xfc) >> 2];
    *buff++ = EncodeTable[((c1 & 0x03) << 4) | ((c2 & 0xf0) >> 4)];
    *buff++ = EncodeTable[((c2 & 0x0f) << 2)];
    *buff++ = '=';
  }

  *len = buff - output;
  return true;
}

bool Base64Decode(const LiteString& input, std::string* output) {
  output->resize(ExpectedDecodeLength(input.size()));

  char* buff = &(*output)[0];
  size_t len = output->size();

  if (!Base64Decode(input, buff, &len)) {
    output->clear();
    return false;
  }
  output->resize(len);
  return true;
}

bool Base64Decode(const LiteString& input, char* output, size_t* len) {
  char* buff = output;
  if (__builtin_expect(*len < ExpectedDecodeLength(input.size()), 0)) {
    return false;
  }
  if (__builtin_expect(!DecodeTable, 0)) {
    FillDecodeTable();
  }
  if (input.empty()) {
    *len = buff - output;
    return true;
  }

  const unsigned char* p = (unsigned char*)input.data();
  const unsigned char* q = (unsigned char*)input.data() + input.size();

  // handle 4 bytes in every loop
  while (true) {
    char ch = 0;
    unsigned char encoded[4];
    int len = GetNext4EncodedCharacters(p, q, encoded);
    if (__builtin_expect(len == 4, 1)) {
      ch =  encoded[0] << 2;  // all 6 bits
      ch |= encoded[1] >> 4;  // 2 high bits
      *buff++ = ch;
      ch =  encoded[1] << 4;  // 4 low bits
      ch |= encoded[2] >> 2;  // 4 high bits
      *buff++ = ch;
      ch =  encoded[2] << 6;  // 2 low bits
      ch |= encoded[3];
      *buff++ = ch;
    } else if (len >= 2) {
      ch =  encoded[0] << 2;  // all 6 bits
      ch |= encoded[1] >> 4;  // 2 high bits
      *buff++ = ch;
      if (len == 3) {
        ch =  encoded[1] << 4;  // 4 low bits
        ch |= encoded[2] >> 2;  // 4 high bits
        *buff++ = ch;
      }
    } else if (len == 0) {
      break;
    } else {
      return false;
    }
  }

  *len = buff - output;
  return true;
}

}  // namespace graphlearn
