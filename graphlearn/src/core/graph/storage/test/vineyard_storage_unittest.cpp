#include <stdio.h>

#include <fstream>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "include/config.h"
#include "core/graph/graph_store.h"

#if defined(WITH_VINEYARD)

#include "vineyard/client/client.h"
#include "vineyard/graph/fragment/arrow_fragment.h"
#include "vineyard/graph/loader/arrow_fragment_loader.h"

#include "core/graph/storage/vineyard_edge_storage.h"
#include "core/graph/storage/vineyard_graph_storage.h"
#include "core/graph/storage/vineyard_node_storage.h"
#include "core/graph/storage/vineyard_storage_utils.h"
#include "core/graph/storage/vineyard_topo_storage.h"
#endif

using namespace graphlearn::io; // NOLINT(build/namespaces)

#if defined(WITH_VINEYARD)

using GraphType = vineyard::ArrowFragment<vineyard_oid_t, vineyard_vid_t>;
using LabelType = typename GraphType::label_id_t;

std::string generate_path(const std::string& prefix, int part_num) {
  if (part_num == 1) {
    return prefix;
  } else {
    std::string ret;
    bool first = true;
    for (int i = 0; i < part_num; ++i) {
      if (first) {
        first = false;
        ret += (prefix + "_" + std::to_string(i));
      } else {
        ret += (";" + prefix + "_" + std::to_string(i));
      }
    }
    return ret;
  }
}

#endif

#if defined(WITH_VINEYARD)
void test_vineyard_storage(vineyard::Client& client,
                          const std::vector<std::string>& efiles,
                          const std::vector<std::string>& vfiles,
                          int directed) {
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  vineyard::ObjectID fragment_group_id = vineyard::InvalidObjectID();
  {
    auto loader = std::unique_ptr<
        vineyard::ArrowFragmentLoader<vineyard_oid_t, vineyard_vid_t>>(
        new vineyard::ArrowFragmentLoader<vineyard_oid_t, vineyard_vid_t>(
            client, comm_spec, efiles, vfiles, directed != 0));
    fragment_group_id = loader->LoadFragmentAsFragmentGroup().value();
  }
  LOG(INFO) << "[worker-" << comm_spec.worker_id()
            << "] loaded graph (fragment group) to vineyard: " << fragment_group_id;

  auto frag =
      std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(client.GetObject(fragment_group_id));

  LOG(INFO) << "obtain graph from vineyard: frag group ptr = " << frag;

  graphlearn::SetGlobalFlagVineyardGraphID(fragment_group_id);
  graphlearn::SetGlobalFlagVineyardIPCSocket("/tmp/vineyard.sock");

  {
    auto store = std::make_shared<VineyardEdgeStorage>("0");
    auto size = store->Size(); // edge size
    LOG(INFO) << "edge size = " << size;
    auto src_ids = store->GetSrcIds();
    auto dst_ids = store->GetDstIds();
    for (int i = 0; i < size; ++i) {
      LOG(INFO) << src_ids[i] << " -> " << dst_ids[i];
    }
  }
  LOG(INFO) << "Passed graph-learn edge storage test...";

  {
    auto store = std::make_shared<VineyardNodeStorage>("0");
    auto size = store->Size(); // edge size
    LOG(INFO) << "node size = " << size;
    auto node_ids = store->GetIds();
    auto label_ids = store->GetLabels();
    auto weights_ids = store->GetWeights();
    for (int i = 0; i < size; ++i) {
      std::stringstream ss;
      ss << node_ids[i];
      if (label_ids) {
        ss << "(" << label_ids[i] << ")";
      }
      if (weights_ids) {
        ss << ": " << weights_ids[i];
      }
      LOG(INFO) << ss.str();
    }
  }
  LOG(INFO) << "Passed graph-learn node storage test...";

  {
    auto store = std::make_shared<VineyardGraphStorage>("0");
    auto size = store->GetEdgeCount(); // edge size
    LOG(INFO) << "edge size = " << size;
    auto src_ids = store->GetAllSrcIds();
    auto dst_ids = store->GetAllDstIds();
    for (size_t idx = 0; idx < src_ids.Size(); ++idx) {
      LOG(INFO) << "src = " << src_ids.at(idx)
                << ", out degree = " << store->GetOutDegree(src_ids.at(idx));
    }
    for (size_t idx = 0; idx < dst_ids.Size(); ++idx) {
      LOG(INFO) << "dst = " << dst_ids.at(idx)
                << ", in degree = " << store->GetInDegree(dst_ids.at(idx));
    }
  }
  LOG(INFO) << "Passed graph-learn graph storage test...";

  {
    auto store = std::make_shared<VineyardTopoStorage>("0");
    auto src_ids = store->GetAllSrcIds();
    for (size_t idx = 0; idx < src_ids.Size(); ++idx) {
      auto src = src_ids.at(idx);
      auto nbrs = store->GetNeighbors(src);
      auto edges = store->GetOutEdges(src);
      CHECK_EQ(nbrs.Size(), edges.Size());
      for (int i = 0; i < nbrs.Size(); ++i) {
        LOG(INFO) << src << " -> " << nbrs[i] << ", edge_id = " << edges[i];
      }
    }
  }
  LOG(INFO) << "Passed graph-learn topo storage test...";
}
#endif

int main(int argc, char **argv) {
if (argc < 6) {
    printf(
        "usage: ./vineyard_storage_unittest <ipc_socket> <e_label_num> <efiles...> "
        "<v_label_num> <vfiles...> [directed]\n");
#if defined(WITH_VINEYARD)
    return 1;
#else
    return 0;
#endif
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  int edge_label_num = atoi(argv[index++]);
  std::vector<std::string> efiles;
  for (int i = 0; i < edge_label_num; ++i) {
    efiles.push_back(argv[index++]);
  }

  int vertex_label_num = atoi(argv[index++]);
  std::vector<std::string> vfiles;
  for (int i = 0; i < vertex_label_num; ++i) {
    vfiles.push_back(argv[index++]);
  }

  int directed = 1;
  if (argc > index) {
    directed = atoi(argv[index]);
  }

#if defined(WITH_VINEYARD)
  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  grape::InitMPIComm();

  {
    test_vineyard_storage(client, efiles, vfiles, directed);
    LOG(INFO) << "Passed graph-learn fragment test...";
  }

  grape::FinalizeMPIComm();

#endif

  return 0;
}

