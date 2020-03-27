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

#ifndef GRAPHLEARN_COMMON_THREADING_ATOMIC_ATOMIC_H_
#define GRAPHLEARN_COMMON_THREADING_ATOMIC_ATOMIC_H_

#include "graphlearn/common/threading/atomic/atomic_detail.h"

namespace graphlearn {

// compare the value stored in 'target' and 'compare', if they are equal,
// then set the 'exchange' into 'target' buffer.
// if the original value in 'target' equals to 'compare', return true
template<typename T>
inline bool AtomicCompareExchange(volatile T* target, T exchange, T compare) {
  return detail::AtomicDetail<sizeof(T)>::CompareExchange(
      target,
      exchange,
      compare);
}

// add 'value' to '*target', and return the original value.
template<typename T>
inline T AtomicExchangeAdd(volatile T* target, T value) {
  return detail::AtomicDetail<sizeof(T)>::ExchangeAdd(target, value);
}

// substract 'value' from '*target', and return the original value.
template<typename T>
inline T AtomicExchangeSub(volatile T* target, T value) {
  return AtomicExchangeAdd(target, static_cast<T>(-value));
}

// add 'value' to '*target', and return the new value in 'target'.
template<typename T>
inline T AtomicAdd(volatile T* target, T value) {
  return AtomicExchangeAdd(target, value) + value;
}

// substract 'value' from '*target', and return the new value in 'target'.
template<typename T>
inline T AtomicSub(volatile T* target, T value) {
  return AtomicExchangeSub(target, value) - value;
}

// set 'value' into 'target', and return the old value in 'target'.
template<typename T>
inline T AtomicExchange(volatile T* target, T value) {
  return detail::AtomicDetail<sizeof(T)>::Exchange(target, value);
}

// set 'value' into 'target'
template<typename T>
inline void AtomicSet(volatile T* target, T value) {
  detail::AtomicDetail<sizeof(T)>::Set(target, value);
}

// get the value in 'target'
template<typename T>
inline T AtomicGet(volatile T* target) {
  T value = *target;
  return value;
}

// add 1 to '*target', and return the new value in 'target'.
template<typename T>
inline T AtomicIncrement(volatile T* target) {
  return AtomicAdd(target, static_cast<T>(1));
}

// substract 1 from '*target', and return the new value in 'target'.
template<typename T>
inline T AtomicDecrement(volatile T* target) {
  return AtomicSub(target, static_cast<T>(1));
}

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_THREADING_ATOMIC_ATOMIC_H_
