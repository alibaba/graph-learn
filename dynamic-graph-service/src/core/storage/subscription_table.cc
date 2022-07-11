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

#include "core/storage/subscription_table.h"

#include "rocksdb/utilities/options_util.h"

#include "common/log.h"
#include "common/options.h"
#include "common/partitioner.h"
#include "core/storage/prefix_extractor.h"

#define LOG_RETURN_IF_BACKUP_NOT_OK(s)       \
  if (!s.ok()) {                             \
    LOG(ERROR) << s.ToString();              \
    return;                                  \
  }                                          \

#define LOG_RETURN_IF_STATUS_NOT_OK(s)  \
  if (!s.ok()) {                        \
    LOG(ERROR) << s.ToString();         \
    return false;                       \
  }                                     \


namespace dgs {
namespace storage {

SubscriptionTable::SubscriptionTable(
    const std::vector<PartitionId>& pids,
    Partitioner&& table_partitioner,
    const std::string& db_path,
    const std::string& backup_path,
    RdbEnv* rdb_env)
  : db_path_(db_path),
    backup_path_(backup_path) {
  Configure(rdb_env);
  bool ok = true;
  for (auto pid : pids) {
    SubsPartitionBackupInfo bk_info(pid);
    ok &= AddPartitionedTable(bk_info);
  }
  SetTablePartitioner(std::move(table_partitioner));
}

SubscriptionTable::SubscriptionTable(
    const std::vector<SubsPartitionBackupInfo>& part_bk_infos,
    Partitioner&& table_partitioner,
    const std::string& db_path,
    const std::string& backup_path,
    RdbEnv* rdb_env)
  : db_path_(db_path),
    backup_path_(backup_path) {
  Configure(rdb_env);
  bool ok = true;
  for (auto& bk_info : part_bk_infos) {
    ok &= AddPartitionedTable(bk_info);
  }
  SetTablePartitioner(std::move(table_partitioner));
}

void SubscriptionTable::Configure(RdbEnv* rdb_env) {
  // TODO(@goldenleaves): using HDFS enabled Env instead.
  backup_env_ = rocksdb::Env::Default();

  auto& gl_opts = Options::GetInstance().GetSubscriptionTableOptions();

  rdb_options_.create_if_missing = true;
  rdb_options_.env = rdb_env;
  // Use prefix_extractor since we don't need total-order iteration.
  rdb_options_.prefix_extractor = std::make_shared<RocksdbPrefixExtractor>();
  rdb_options_.max_open_files = -1;
  rdb_options_.max_background_jobs = 4;
  rdb_options_.compression = rocksdb::kLZ4Compression;

  // TODO(@goldenleaves): configure bloom filter.
  rocksdb::BlockBasedTableOptions table_options;
  table_options.index_type = rocksdb::BlockBasedTableOptions::kHashSearch;
  table_options.cache_index_and_filter_blocks = true;
  table_options.pin_l0_filter_and_index_blocks_in_cache = true;
  // TODO(@goldenleaves): set num_shards_bit
  table_options.block_cache = rocksdb::NewLRUCache(
      gl_opts.block_cache_capacity);
  rdb_options_.table_factory.reset(
      rocksdb::NewBlockBasedTableFactory(table_options));

  if (gl_opts.memtable_rep == "hashskiplist") {
    rdb_options_.memtable_factory.reset(rocksdb::NewHashLinkListRepFactory(
        gl_opts.hash_bucket_count));
    rdb_options_.allow_concurrent_memtable_write = false;
  } else if (gl_opts.memtable_rep == "skiplist") {
    rdb_options_.memtable_factory.reset(new rocksdb::SkipListFactory(
        gl_opts.skip_list_lookahead));
    rdb_options_.allow_concurrent_memtable_write = true;
  } else {
    LOG(FATAL) << "Unsupported memtable rep: " << gl_opts.memtable_rep;
  }

  // disable Write-Ahead-Log.
  write_options_.disableWAL = true;
  // only iterate in a prefix-same range.
  read_options_.prefix_same_as_start = true;

  ttl_in_seconds_ = static_cast<int32_t>(gl_opts.ttl_in_hours) * 3600;

  LOG(INFO) << "RdbEnv #HighPriorityBackgroundThreads is "
            << rdb_options_.env->GetBackgroundThreads(RdbEnv::Priority::HIGH);
  LOG(INFO) << "RdbEnv #LowPriorityBackgroundThreads is "
            << rdb_options_.env->GetBackgroundThreads(RdbEnv::Priority::LOW);
}

SubscriptionTable::~SubscriptionTable() {
  bool ok = true;
  for (uint32_t i = 0; i < table_insts_.size(); ++i) {
    ok &= RemovePartitionedTable(i);
  }

  if (backup_env_ != rocksdb::Env::Default()) {
    delete backup_env_;
    backup_env_ = nullptr;
  }
}

bool SubscriptionTable::GetSubscribedWorkers(
    uint32_t index,
    const io::SubsRule::Prefix& rule_prefix,
    std::vector<SubsInfo>* output) {
  if (IsManagedOp(rule_prefix.op_id)) {
    output->push_back(
      {index, dsw_partitioner_.GetPartitionId(rule_prefix.vid)});
    return true;
  } else {
    auto pid = tbl_partitioner_.GetPartitionId(rule_prefix.vid);
    auto *db = table_insts_[pid].db;
    auto *iter = db->NewIterator(read_options_);

    const auto prefix_slice = rule_prefix.ToSlice();
    iter->Seek(rocksdb::Slice{prefix_slice.data(), prefix_slice.size()});
    bool is_subscribed = iter->Valid();
    while (iter->Valid()) {
      auto *rule = reinterpret_cast<const io::SubsRule*>(iter->key().data());
      output->push_back({index, static_cast<WorkerId>(rule->worker_id)});
      iter->Next();
    }
    delete iter;
    return is_subscribed;
  }
}

bool SubscriptionTable::UpdateRules(
    const std::vector<io::SubsRule>& rules,
    std::vector<uint32_t>* output) {
  bool ret = false;
  std::string value_buffer;
  for (uint32_t i = 0; i < rules.size(); ++i) {
    auto &rule = rules[i];
    assert(!IsManagedOp(rule.pkey.op_id));
    const auto key_slice = rule.ToSlice();
    auto pid = tbl_partitioner_.GetPartitionId(rule.pkey.vid);
    auto *db = table_insts_[pid].db;
    auto s = db->Get(read_options_, {key_slice.data(), key_slice.size()},
                      &value_buffer);
    if (!s.ok()) {
      // we are adding a new rule.
      output->push_back(i);
    }
    s = db->Put(write_options_, {key_slice.data(), key_slice.size()}, {});
    ret &= s.ok();
  }
  return ret;
}

void SubscriptionTable::Init(const execution::Dag* dag) {
  static bool is_inited = false;
  std::lock_guard<std::mutex> guard(subs_table_mtx_);
  if (is_inited) {
    LOG(INFO) << "SubscriptionTable is already inited.";
    return;
  }

  auto *root_node = dag->root();
  if (root_node->id() != 0 || root_node->kind() != PlanNode::Kind_SOURCE) {
    LOG(FATAL) << "Root node's id must be 0 and its kind must be SOURCE";
  }

  // root DagNode's downstreams are managed ops.
  for (auto *edge : root_node->out_edges()) {
    managed_ops_.insert(edge->dst()->id());
  }

  is_inited = true;
  LOG(INFO) << "SubscriptionTable is inited by query plan.";
}

void SubscriptionTable::BackupPartitionedTable(SubsPartitionBackupInfo& bk_info) {
  auto pid = bk_info.pid;
  if (pid >= table_insts_.size() || !table_insts_[pid].is_open) {
    // This partition is not belong to current worker, skip.
    return;
  }

  rocksdb::IOStatus s;
  rocksdb::BackupID backup_id;
  rocksdb::CreateBackupOptions options;
  // Since write-ahead logs are disabled
  options.flush_before_backup = true;

  auto &pdb = table_insts_[pid];
  s = pdb.be->CreateNewBackup(options, pdb.db, &backup_id);
  LOG_RETURN_IF_BACKUP_NOT_OK(s);
  s = pdb.be->VerifyBackup(backup_id);
  LOG_RETURN_IF_BACKUP_NOT_OK(s);

  bk_info.bid = backup_id;
  bk_info.valid = true;
}

std::vector<SubsPartitionBackupInfo> SubscriptionTable::Backup() {
  // Note that it may take a long time for this function to return
  auto num_part = tbl_partitioner_.GetPartitionsNum();
  std::vector<SubsPartitionBackupInfo> sub_bk_infos;
  for (PartitionId pid = 0; pid < num_part; ++pid) {
    SubsPartitionBackupInfo bk_info(pid, false);
    BackupPartitionedTable(bk_info);
    if (bk_info.valid) {
      sub_bk_infos.emplace_back(bk_info);
    }
  }
  return sub_bk_infos;
}

void SubscriptionTable::SetDSWorkerPartitioner(uint32_t partition_num,
                                               const std::string& strategy) {
  try {
    dsw_partitioner_ = PartitionerFactory::Create(strategy, partition_num);
    LOG(INFO) << "Set downstream worker partitioner with strategy: "
              << strategy << ", #workers: " << partition_num;
  } catch (std::exception& ex) {
    LOG(ERROR) << "Set downstream worker partitioner with stragety: "
               << strategy << ", #workers: " << partition_num
               << " failed: " << ex.what();
  }
}

void SubscriptionTable::SetTablePartitioner(Partitioner&& partitioner) {
  LOG(INFO) << "New table partitioner is set.";
  tbl_partitioner_ = std::move(partitioner);
}

bool SubscriptionTable::AddPartitionedTable(const SubsPartitionBackupInfo& bk_info) {
  auto pid = bk_info.pid;
  auto is_restore = bk_info.valid;
  std::lock_guard<std::mutex> guard(subs_table_mtx_);
  if (pid < table_insts_.size() && table_insts_[pid].is_open) {
    LOG(WARNING) << "SubscriptionTable Partition " << pid << " already exists."
                 << " DB path is " << db_path_;
    return true;
  }

  if (pid >= table_insts_.size()) {
    // expand vector size.
    table_insts_.resize(pid + 1);
  }

  PartitionedTable ptable;
  bool ok;
  if (is_restore) {
    LOG(INFO) << "Restoring SubscriptionTable Partition " << pid
              << ". db backup id: " << bk_info.bid;
    ok = ptable.Restore(bk_info, db_path_, backup_path_, rdb_options_,
                        backup_env_, ttl_in_seconds_);
  }
  else {
    ok = ptable.Open(pid, db_path_, backup_path_, rdb_options_,
                     backup_env_, ttl_in_seconds_);
  }
  
  if (ok) {
    table_insts_[pid] = std::move(ptable);
    LOG(INFO) << "SubscriptionTable Partition " << pid << " is added."
              << " DB path is " << db_path_;
  } else {
    LOG(ERROR) << "SubscriptionTable Partition " << pid << " add failed."
               << " DB path is " << db_path_;
  }
  return ok;
}

bool SubscriptionTable::RemovePartitionedTable(PartitionId pid) {
  std::lock_guard<std::mutex> guard(subs_table_mtx_);
  if (pid >= table_insts_.size() || !table_insts_[pid].is_open) {
    LOG(WARNING) << "SubscriptionTable Partition " << pid << " doesn't exist."
                 << " DB path is " << db_path_;
    return true;
  }

  bool ok = table_insts_[pid].Close();
  if (ok) {
    table_insts_[pid] = PartitionedTable();
  } else {
    LOG(ERROR) << "SubscriptionTable Partition " << pid << " Close() failed."
               << " DB path is " << db_path_;
  }
  return ok;
}

bool SubscriptionTable::PartitionedTable::Open(PartitionId partition_id,
                                               const std::string& db_path,
                                               const std::string& backup_path,
                                               const rocksdb::Options& options,
                                               rocksdb::Env* backup_env,
                                               uint32_t ttl_in_seconds) {
  // TODO(@goldenleaves): check whether db_path&backup_path is end with '/'
  auto db_name = db_path + "/subs_table_part_" + std::to_string(partition_id);
  auto be_name = backup_path + "/subs_table_bak_part_"
                              + std::to_string(partition_id);

  auto s = rocksdb::DBWithTTL::Open(options, db_name, &db, ttl_in_seconds);
  LOG_RETURN_IF_STATUS_NOT_OK(s)
  auto io_s = rocksdb::BackupEngine::Open(
      backup_env,rocksdb::BackupEngineOptions(be_name), &be);
  LOG_RETURN_IF_STATUS_NOT_OK(io_s)

  is_open = true;
  return true;
}

bool SubscriptionTable::PartitionedTable::Restore(const SubsPartitionBackupInfo& bk_info,
                                                  const std::string& db_path,
                                                  const std::string& backup_path,
                                                  const rocksdb::Options& options,
                                                  rocksdb::Env* backup_env,
                                                  uint32_t ttl_in_seconds) {
  auto partition_id = bk_info.pid;
  auto db_name = db_path + "/subs_table_part_" + std::to_string(partition_id);
  auto be_name = backup_path + "/subs_table_bak_part_"
                              + std::to_string(partition_id);

  // Open backup engine
  auto backup_status = rocksdb::BackupEngine::Open(backup_env,
            rocksdb::BackupEngineOptions(be_name), &be);
  LOG_RETURN_IF_STATUS_NOT_OK(backup_status)

  // Restore DB from backup with the specified backup id
  backup_status = be->RestoreDBFromBackup(bk_info.bid, db_name, db_name);
  LOG_RETURN_IF_STATUS_NOT_OK(backup_status)

  // Open DB
  auto open_status = rocksdb::DBWithTTL::Open(options, db_name, &db, ttl_in_seconds);
  LOG_RETURN_IF_STATUS_NOT_OK(open_status)

  is_open = true;
  return true;
}

bool SubscriptionTable::PartitionedTable::Close() {
  auto s = db->Close();
  LOG_RETURN_IF_STATUS_NOT_OK(s);

  is_open = false;
  // TODO(@goldenleaves): cleanup backup engine.
  return true;
}

}  // namespace storage
}  // namespace dgs
