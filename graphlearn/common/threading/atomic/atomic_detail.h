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

#ifndef GRAPHLEARN_COMMON_THREADING_ATOMIC_ATOMIC_DETAIL_H_
#define GRAPHLEARN_COMMON_THREADING_ATOMIC_ATOMIC_DETAIL_H_

#include <cstdint>

namespace graphlearn {
namespace detail {

class AtomicDetailDefault {
public:
  template<typename T>
  static void Set(volatile T* target, T value) {
    *target = value;
  }

  template<typename T>
  static bool CompareExchange(volatile T* target, T exchange, T compare) {
    return __sync_bool_compare_and_swap(target, compare, exchange);
  }

  template<typename T>
  static T ExchangeAdd(volatile T* target, T value) {
    return __sync_fetch_and_add(target, value);
  }

  template<typename T>
  static T Exchange(volatile T* target, T value) {
    return __sync_lock_test_and_set(target, value);
  }
};

template<int Size>
class AtomicDetail {
public:
  template<typename T>
  static void Set(volatile T* target, T value);
  template<typename T>
  static bool CompareExchange(volatile T* target, T exchange, T compare);
  template<typename T>
  static T ExchangeAdd(volatile T* target, T value);
  template<typename T>
  static T Exchange(volatile T* target, T value);
};

template<>
class AtomicDetail<1> : public AtomicDetailDefault {
};

template<>
class AtomicDetail<2> : public AtomicDetailDefault {
};

template<>
class AtomicDetail<4> : public AtomicDetailDefault {
};

#if __i386__

template<>
class AtomicDetail<8> : public AtomicDetailDefault {
public:
  template<typename T>
  static void Set(volatile T* target, T value) {
    __sync_lock_test_and_set(target, value);
  }
};

#elif __x86_64__

template<>
class AtomicDetail<8> : public AtomicDetailDefault {
};

template<>
class AtomicDetail<16> {
public:
  template<typename T>
  static bool CompareExchange(volatile T* target, T exchange, T compare) {
    uint64_t *cmp = reinterpret_cast<uint64_t*>(&compare);
    uint64_t *with = reinterpret_cast<uint64_t*>(&exchange);
    bool result;
    __asm__ __volatile__ (
         "lock cmpxchg16b %1\n\t"
         "setz %0"
         : "=q" (result), "+m" (*target), "+d" (cmp[1]), "+a" (cmp[0])
         : "c" (with[1]), "b" (with[0])
         : "cc" );
    return result;
  }
};

#else

#error ARCHITECT NOT SUPPORTED

#endif

}  // namespace detail
}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_THREADING_ATOMIC_ATOMIC_DETAIL_H_
