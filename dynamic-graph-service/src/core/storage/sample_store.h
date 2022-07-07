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

#ifndef DGS_CORE_STORAGE_SAMPLE_STORE_H_
#define DGS_CORE_STORAGE_SAMPLE_STORE_H_

#include <functional>
#include <mutex>
#include <unordered_map>

#include "rocksdb/utilities/db_ttl.h"
#include "rocksdb/utilities/backupable_db.h"

#include "common/partitioner.h"
#include "core/storage/env.h"
#include "core/storage/key.h"
#include "core/io/record.h"
#include "core/io/record_slice.h"

namespace dgs {

class Service;

namespace storage {

struct KVPair {
  Key        key;
  io::Record value;

  KVPair(const Key& k, io::Record&& v)
    : key(k), value(std::move(v)) {}
};

struct StorePartitionBackupInfo {
  PartitionId        pid;
  bool               valid; // false if either vertex_bid or edge_bid is not valid
  PartitionBackupId  vertex_bid = 0;
  PartitionBackupId  edge_bid = 0;

  StorePartitionBackupInfo(const PartitionId& pid,
                           const PartitionBackupId& v_bid,
                           const PartitionBackupId& e_bid,
                           bool valid = true)
    : pid(pid), valid(valid), vertex_bid(v_bid),  edge_bid(e_bid) {}

  explicit StorePartitionBackupInfo(const PartitionId& pid, bool valid = false)
    : pid(pid), valid(valid) {}
};

class SampleStore {
public:
  SampleStore(const std::vector<PartitionId>& pids,
              Partitioner&& partitioner,
              const std::string& db_path,
              const std::string& backup_path,
              RdbEnv* rdb_env);
  SampleStore(const std::vector<StorePartitionBackupInfo>& part_bk_infos,
              Partitioner&& partitioner,
              const std::string& db_path,
              const std::string& backup_path,
              RdbEnv* rdb_env);

  SampleStore(SampleStore&& other) = delete;
  SampleStore(const SampleStore& other) = delete;

  ~SampleStore();

  // Read APIs.
  bool GetVertex(const Key& key, io::Record* record) const;
  bool GetEdge(const Key& key, io::Record* record) const;

  bool GetVerticesByPrefix(const Key::Prefix& prefix_key,
                           std::vector<KVPair>* records) const;
  bool GetEdgesByPrefix(const Key::Prefix& prefix_key,
                        std::vector<KVPair>* records) const;

  /// Write APIs.
  bool PutVertex(const Key& key, const io::RecordSlice& slice);
  bool PutEdge(const Key& key, const io::RecordSlice& slice);

  bool DeleteVertex(const Key& key);
  bool DeleteEdge(const Key& key);

  bool DeleteVerticesByPrefix(const Key::Prefix& prefix_key);
  bool DeleteEdgesByPrefix(const Key::Prefix& prefix_key);

  /// DB backup APIs.
  std::vector<StorePartitionBackupInfo> Backup();

private:
  // DB management APIs.
  bool AddPartitionedDB(const StorePartitionBackupInfo& bk_info);
  bool RemovePartitionedDB(PartitionId pid);

  void BackupPartitionedDB(StorePartitionBackupInfo& bk_info);

  void SetPartitioner(Partitioner&& partitioner);
  size_t GetLocalPartitionNum() const;

private:
  void Configure(RdbEnv* rdb_env);

private:
  struct PartitionedDB {
    PartitionedDB() : vtx_db(nullptr), vtx_be(nullptr),
        edge_db(nullptr), edge_be(nullptr), is_open(false) {
    }

    PartitionedDB(PartitionedDB&& other);
    PartitionedDB& operator=(PartitionedDB&& other);

    bool Open(PartitionId partition_id,
              const std::string& db_path,
              const std::string& backup_path,
              const rocksdb::Options& options,
              rocksdb::Env* backup_env,
              uint32_t ttl_in_seconds);
    bool Restore(const StorePartitionBackupInfo& bk_info,
                 const std::string& db_path,
                 const std::string& backup_path,
                 const rocksdb::Options& options,
                 rocksdb::Env* backup_env,
                 uint32_t ttl_in_seconds);

    bool Close();

    rocksdb::DB           *vtx_db;
    rocksdb::BackupEngine *vtx_be;
    rocksdb::DBWithTTL    *edge_db;
    rocksdb::BackupEngine *edge_be;
    bool                  is_open;
  };

  const std::string          db_path_;
  const std::string          backup_path_;
  rocksdb::Env*              backup_env_;
  rocksdb::WriteOptions      write_options_;
  rocksdb::ReadOptions       read_options_;
  rocksdb::Options           rdb_options_;
  uint32_t                   local_part_num_;
  uint32_t                   ttl_in_seconds_;
  // using vector instead of unordered_map here
  // to optimize real space/time complexity.
  std::vector<PartitionedDB> db_insts_;
  std::mutex                 db_insts_mtx_;
  Partitioner                partitioner_;

  friend class dgs::Service;
};

}  // namespace storage
}  // namespace dgs

#endif  // DGS_CORE_STORAGE_SAMPLE_STORE_H_
