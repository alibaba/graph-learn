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

#ifndef GRAPHLEARN_COMMON_THREADING_SYNC_LOCK_H_
#define GRAPHLEARN_COMMON_THREADING_SYNC_LOCK_H_

#include <pthread.h>
#include <mutex>  // NOLINT [build/c++11]
#include "graphlearn/common/base/uncopyable.h"
#include "graphlearn/common/threading/atomic/atomic.h"
#include "graphlearn/common/threading/this_thread.h"

namespace graphlearn {

template<typename T>
class ScopedLocker : private Uncopyable {
public:
  using LockType = T;

  explicit ScopedLocker(T* lock)
    : lock_(lock) {
    lock_->Lock();
  }
  explicit ScopedLocker(T& lock)  // NOLINT [runtime/references]
    : lock_(&lock) {
    lock_->Lock();
  }

  ~ScopedLocker() {
    lock_->Unlock();
  }

private:
  T* lock_;
};

template<>
class ScopedLocker<pthread_mutex_t> : private Uncopyable {
public:
  using T = pthread_mutex_t;
  using LockType = T;

  explicit ScopedLocker(T* lock)
    : lock_(lock) {
    ::pthread_mutex_lock(lock_);
  }

  explicit ScopedLocker(T& lock)  // NOLINT [runtime/references]
    : lock_(&lock) {
    ::pthread_mutex_lock(lock_);
  }

  ~ScopedLocker() {
    ::pthread_mutex_unlock(lock_);
  }

private:
  pthread_mutex_t* lock_;
};

template <>
class ScopedLocker<std::mutex> : private Uncopyable {
public:
  using T = std::mutex;

  explicit ScopedLocker(T* lock)
    : lock_(lock) {
    lock_->lock();
  }

  ~ScopedLocker() {
    lock_->unlock();
  }

private:
  T* lock_;
};

template<typename T>
class ScopedUnlocker : private Uncopyable {
public:
  using LockType = T;

  explicit ScopedUnlocker(T* lock)
    : lock_(lock) {
    lock_->Unlock();
  }

  explicit ScopedUnlocker(T& lock)  // NOLINT [runtime/references]
    : lock_(&lock) {
    lock_->Unlock();
  }

  ~ScopedUnlocker() {
    lock_->Lock();
  }

private:
  T* lock_;
};

class NullLock : private Uncopyable {
public:
  using Locker = ScopedLocker<NullLock>;
  using Unlocker = ScopedUnlocker<NullLock>;

  NullLock() : locked_(false) { }

  ~NullLock() { }

  void Lock() {
    locked_ = true;
  }

  void Unlock() {
    locked_ = false;
  }

  bool IsLocked() const {
    return locked_;
  }

private:
  bool locked_;
};

class SpinLock : private Uncopyable {
public:
  using Locker = ScopedLocker<SpinLock>;
  using Unlocker = ScopedUnlocker<SpinLock>;

  SpinLock() {
    ::pthread_spin_init(&spin_, PTHREAD_PROCESS_PRIVATE);
  }

  ~SpinLock() {
    ::pthread_spin_destroy(&spin_);
  }

  void Lock() {
    ::pthread_spin_lock(&spin_);
  }

  void Unlock() {
    ::pthread_spin_unlock(&spin_);
  }

  bool IsLocked() const {
    return spin_ == 0;
  }

private:
  pthread_spinlock_t spin_;
};

class MutexBase : private Uncopyable {
protected:
  explicit MutexBase(int type) {
    pthread_mutexattr_t attr;
    ::pthread_mutexattr_init(&attr);
    ::pthread_mutexattr_settype(&attr, type);
    ::pthread_mutex_init(&mutex_, &attr);
    ::pthread_mutexattr_destroy(&attr);
  }

public:
  virtual ~MutexBase() {
    ::pthread_mutex_destroy(&mutex_);
  }

  void Lock() {
    ::pthread_mutex_lock(&mutex_);
  }

  void Unlock() {
    ::pthread_mutex_unlock(&mutex_);
  }

  bool IsLocked() const {
    return mutex_.__data.__lock > 0;
  }

  pthread_mutex_t* NativeLock() {
    return &mutex_;
  }

protected:
  pthread_mutex_t mutex_;
};

class SimpleMutex : public MutexBase {
public:
  using Locker = ScopedLocker<SimpleMutex>;
  using Unlocker = ScopedUnlocker<SimpleMutex>;

  SimpleMutex() : MutexBase(PTHREAD_MUTEX_NORMAL) { }
};

class RestrictMutex : public MutexBase {
public:
  using Locker = ScopedLocker<RestrictMutex>;
  using Unlocker = ScopedUnlocker<RestrictMutex>;

  RestrictMutex() : MutexBase(PTHREAD_MUTEX_ERRORCHECK) { }
};

class RecursiveMutex : public MutexBase {
public:
  using Locker = ScopedLocker<RecursiveMutex>;
  using Unlocker = ScopedUnlocker<RecursiveMutex>;

  RecursiveMutex() : MutexBase(PTHREAD_MUTEX_RECURSIVE) { }
};

class AdaptiveMutex : public MutexBase {
public:
  using Locker = ScopedLocker<AdaptiveMutex>;
  using Unlocker = ScopedUnlocker<AdaptiveMutex>;

  AdaptiveMutex() : MutexBase(PTHREAD_MUTEX_ADAPTIVE_NP) { }
};

// read write lock
template <typename T>
class ScopedReaderLocker : private Uncopyable {
public:
  using LockType = T;

  explicit ScopedReaderLocker(T* lock)
    : lock_(lock) {
    lock_->ReadLock();
  }

  explicit ScopedReaderLocker(T& lock)  // NOLINT [runtime/references]
    : lock_(&lock) {
    lock_->ReadLock();
  }

  ~ScopedReaderLocker() {
    lock_->ReadUnlock();
  }

private:
  T* lock_;
};

template <typename T>
class ScopedWriterLocker : private Uncopyable {
public:
  using LockType = T;

  explicit ScopedWriterLocker(T* lock)
    : lock_(lock) {
    lock_->WriteLock();
  }

  explicit ScopedWriterLocker(T& lock)  // NOLINT [runtime/references]
    : lock_(&lock) {
    lock_->WriteLock();
  }

  ~ScopedWriterLocker() {
    lock_->WriteUnlock();
  }

private:
  T* lock_;
};

class RWLock : private Uncopyable {
public:
  using ReaderLocker = ScopedReaderLocker<RWLock>;
  using WriterLocker = ScopedWriterLocker<RWLock>;

  enum Mode {
    kModePreferReader = PTHREAD_RWLOCK_PREFER_READER_NP,
    kModePreferWriter = PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP,
    kModeDefault = PTHREAD_RWLOCK_DEFAULT_NP,
    PREFER_READER = kModePreferReader,
    PREFER_WRITER = kModePreferWriter
  };

  explicit RWLock(Mode mode = kModeDefault) {
    pthread_rwlockattr_t attr;
    ::pthread_rwlockattr_init(&attr);
    ::pthread_rwlockattr_setkind_np(&attr, mode);
    ::pthread_rwlock_init(&lock_, &attr);
    ::pthread_rwlockattr_destroy(&attr);
  }

  ~RWLock() {
    ::pthread_rwlock_destroy(&lock_);
  }

  void ReadLock() {
    ::pthread_rwlock_rdlock(&lock_);
  }

  void WriteLock() {
    ::pthread_rwlock_wrlock(&lock_);
  }

  bool TryReadLock() {
    return ::pthread_rwlock_tryrdlock(&lock_);
  }

  bool TryWriteLock() {
    return ::pthread_rwlock_trywrlock(&lock_);
  }

  void ReadUnlock() {
    Unlock();
  }

  void WriteUnlock() {
    Unlock();
  }

  void Unlock() {
    ::pthread_rwlock_unlock(&lock_);
  }

private:
  pthread_rwlock_t lock_;
};

// This is a light spin read-write lock, the idea is from the linux kernel.
// It's read perfered, and it's a spin lock! only use it if you are sure in
// you cases, it could lock success quickly, and the punishment is acceptable.
// for detail information, please search 'read_lock' or 'write_lock' in
// linux kernel source code (/include/linux/rwlock.h).
class SpinRWLock : private Uncopyable {
public:
  using ReaderLocker = ScopedReaderLocker<SpinRWLock>;
  using WriterLocker = ScopedWriterLocker<SpinRWLock>;

  SpinRWLock() {
    AtomicSet<int>(&quota_, kInitQuota);
  }

  ~SpinRWLock() { }

  void ReadLock() {
    while (!TryReadLock()) {
      ThisThread::Yield();
    }
  }

  void WriteLock() {
    while (!TryWriteLock()) {
      ThisThread::Yield();
    }
  }

  bool TryReadLock() {
    if (AtomicSub<int>(&quota_, kRead) >= 0) {
      return true;
    } else {
      AtomicAdd<int>(&quota_, kRead);
      return false;
    }
  }

  bool TryWriteLock() {
    if (AtomicSub<int>(&quota_, kWrite) >= 0) {
      return true;
    } else {
      AtomicAdd<int>(&quota_, kWrite);
      return false;
    }
  }

  void ReadUnlock() {
    AtomicAdd<int>(&quota_, kRead);
  }

  void WriteUnlock() {
    AtomicAdd<int>(&quota_, kWrite);
  }

private:
  enum {
    kInitQuota = 0x01000000,
    kWrite = 0x01000000,
    kRead = 0x00000001,
  };

  int quota_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_COMMON_THREADING_SYNC_LOCK_H_
