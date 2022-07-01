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

#ifndef DGS_CORE_STORAGE_SUBSCRIPTION_TABLE_H_
#define DGS_CORE_STORAGE_SUBSCRIPTION_TABLE_H_

#include "rocksdb/utilities/db_ttl.h"
#include "rocksdb/utilities/backupable_db.h"

#include "common/typedefs.h"
#include "core/execution/dag.h"
#include "core/io/subscription_rule.h"
#include "core/storage/env.h"
#include "core/storage/key.h"
#include "core/storage/sample_store.h"

namespace dgs {
namespace storage {

struct SubsInfo {
  uint32_t record_id;
  WorkerId worker_id;

  SubsInfo(uint32_t rid, WorkerId wid)
    : record_id(rid), worker_id(wid) {}
};

struct SubsPartitionBackupInfo {
  PartitionId        pid;
  bool               valid; // false if bid is not valid
  PartitionBackupId  bid = 0;

  SubsPartitionBackupInfo(const PartitionId& pid,
                          const PartitionBackupId& bid,
                          bool valid = true)
    : pid(pid), valid(valid), bid(bid) {}

  explicit SubsPartitionBackupInfo(const PartitionId& pid, bool valid = false)
    : pid(pid), valid(valid) {}
};

class SubscriptionTable {
public:
  SubscriptionTable(const std::vector<PartitionId>& pids,
                    Partitioner&& table_partitioner,
                    const std::string& db_path,
                    const std::string& backup_path,
                    RdbEnv* rdb_env);
  SubscriptionTable(const std::vector<SubsPartitionBackupInfo>& part_bk_infos,
                    Partitioner&& table_partitioner,
                    const std::string& db_path,
                    const std::string& backup_path,
                    RdbEnv* rdb_env);
  ~SubscriptionTable();

  bool GetSubscribedWorkers(
    uint32_t index,
    const io::SubsRule::Prefix& rule_prefix,
    std::vector<SubsInfo>* output);

  bool UpdateRules(
    const std::vector<io::SubsRule>& rules,
    std::vector<uint32_t>* output);

  void SetDSWorkerPartitioner(
    uint32_t partition_num,
    const std::string& strategy);

  void Init(const execution::Dag* dag);
  std::vector<SubsPartitionBackupInfo> Backup();

private:
  void Configure(RdbEnv* rdb_env);

  bool AddPartitionedTable(const SubsPartitionBackupInfo& bk_info);
  bool RemovePartitionedTable(PartitionId pid);
  void SetTablePartitioner(Partitioner&& partitioner);
  void BackupPartitionedTable(SubsPartitionBackupInfo& bk_info);

private:
  bool IsManagedOp(OperatorId op_id) const {
    return managed_ops_.count(op_id);
  }

private:
  struct PartitionedTable {
    PartitionedTable()
      : db(nullptr), be(nullptr),
        is_open(false) {}

    bool Open(PartitionId partition_id,
              const std::string& db_path,
              const std::string& backup_path,
              const rocksdb::Options& options,
              rocksdb::Env* backup_env,
              uint32_t ttl_in_seconds);
    bool Restore(const SubsPartitionBackupInfo& bk_info,
                 const std::string& db_path,
                 const std::string& backup_path,
                 const rocksdb::Options& options,
                 rocksdb::Env* backup_env,
                 uint32_t ttl_in_seconds);
    bool Close();

    PartitionedTable(PartitionedTable&&) = default;
    PartitionedTable& operator=(PartitionedTable&&) = default;

    rocksdb::DBWithTTL    *db;
    rocksdb::BackupEngine *be;
    bool is_open;
  };

private:
  const std::string          db_path_;
  const std::string          backup_path_;
  rocksdb::Env*              backup_env_;
  rocksdb::WriteOptions      write_options_;
  rocksdb::ReadOptions       read_options_;
  rocksdb::Options           rdb_options_;
  uint32_t                   ttl_in_seconds_;
  std::vector<PartitionedTable> table_insts_;
  std::mutex                    subs_table_mtx_;
  Partitioner                   tbl_partitioner_;
  // downstream(serving link) partitioner
  // FIXME(@xmqin): worker granularity.
  Partitioner                    dsw_partitioner_;
  std::unordered_set<OperatorId> managed_ops_;
};

}  // namespace storage
}  // namespace dgs

#endif  // DGS_CORE_STORAGE_SUBSCRIPTION_TABLE_H_
