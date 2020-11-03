#include <stdio.h>

#include <fstream>
#include <string>

#include "glog/logging.h"

#include "vineyard/client/client.h"
#include "vineyard/graph/fragment/arrow_fragment.h"
#include "vineyard/graph/loader/arrow_fragment_loader.h"

#include "graphlearn/include/config.h"
#include "graphlearn/core/graph/storage/vineyard_edge_storage.h"
#include "graphlearn/core/graph/storage/vineyard_graph_storage.h"
#include "graphlearn/core/graph/storage/vineyard_node_storage.h"
#include "graphlearn/core/graph/storage/vineyard_storage_utils.h"
#include "graphlearn/core/graph/storage/vineyard_topo_storage.h"

using namespace graphlearn::io; // NOLINT(build/namespaces)

#if defined(WITH_VINEYARD)

using GraphType = vineyard::ArrowFragment<vineyard_oid_t, vineyard_vid_t>;
using LabelType = typename GraphType::label_id_t;

std::string generate_path(const std::string &prefix, int part_num) {
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

int main(int argc, char **argv) {
  if (argc < 8) {
    printf("usage: ./graph_learn_test <ipc_socket> "
           "<efile_prefix> <e_label_num> <efile_part> "
           "<vfile_prefix> <v_label_num> <vfile_part> [directed]\n");
    return 1;
  }

#if defined(WITH_VINEYARD)

  std::string ipc_socket = std::string(argv[1]);
  std::string epath = generate_path(argv[2], atoi(argv[4]));
  std::string vpath = generate_path(argv[5], atoi(argv[7]));
  int edge_label_num = atoi(argv[3]);
  int vertex_label_num = atoi(argv[6]);
  int directed = 1;
  if (argc >= 9) {
    directed = atoi(argv[8]);
  }

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  grape::InitMPIComm();
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  vineyard::ObjectID fragment_id = vineyard::InvalidObjectID();
  {
    auto loader = std::unique_ptr<
        vineyard::ArrowFragmentLoader<vineyard_oid_t, vineyard_vid_t>>(
        new vineyard::ArrowFragmentLoader<vineyard_oid_t, vineyard_vid_t>(
            client, comm_spec, vertex_label_num, edge_label_num, epath, vpath,
            directed != 0));
    fragment_id = loader->LoadFragment().value();
  }
  LOG(INFO) << "[worker-" << comm_spec.worker_id()
            << "] loaded graph to vineyard: " << fragment_id;

  auto frag =
      std::dynamic_pointer_cast<GraphType>(client.GetObject(fragment_id));

  LOG(INFO) << "obtain graph from vineyard: frag ptr = " << frag;
  
  graphlearn::SetGlobalFlagVineyardGraphID(fragment_id);
  graphlearn::SetGlobalFlagVineyardIPCSocket("/tmp/vineyard.sock");

  {
    auto store = std::make_shared<VineyardEdgeStorage>(0);
    auto size = store->Size(); // edge size
    LOG(INFO) << "edge size = " << size;
    auto &src_ids = *store->GetSrcIds();
    auto &dst_ids = *store->GetDstIds();
    auto &weights_ids = *store->GetWeights();
    for (int i = 0; i < size; ++i) {
      LOG(INFO) << src_ids[i] << " -> " << dst_ids[i] << ": " << weights_ids[i];
    }
  }
  LOG(INFO) << "Passed graph-learn edge storage test...";

  {
    auto store = std::make_shared<VineyardNodeStorage>(0);
    auto size = store->Size(); // edge size
    LOG(INFO) << "node size = " << size;
    auto &node_ids = *store->GetIds();
    auto &label_ids = *store->GetLabels();
    auto &weights_ids = *store->GetWeights();
    for (int i = 0; i < size; ++i) {
      std::stringstream ss;
      ss << node_ids[i];
      if (&label_ids != nullptr) {
        ss << "(" << label_ids[i] << ")";
      }
      if (&weights_ids != nullptr) {
        ss << ": " << weights_ids[i];
      }
      LOG(INFO) << ss.str();
    }
  }
  LOG(INFO) << "Passed graph-learn node storage test...";

  {
    auto store = std::make_shared<VineyardGraphStorage>(0);
    auto size = store->GetEdgeCount(); // edge size
    LOG(INFO) << "edge size = " << size;
    auto &src_ids = *store->GetAllSrcIds();
    auto &dst_ids = *store->GetAllDstIds();
    for (auto const &src : src_ids) {
      LOG(INFO) << "src = " << src
                << ", out degree = " << store->GetOutDegree(src);
    }
    for (auto const &dst : dst_ids) {
      LOG(INFO) << "dst = " << dst
                << ", in degree = " << store->GetInDegree(dst);
    }
  }
  LOG(INFO) << "Passed graph-learn graph storage test...";

  {
    auto store = std::make_shared<VineyardTopoStorage>(0);
    auto &src_ids = *store->GetAllSrcIds();
    for (auto const &src : src_ids) {
      auto nbrs = store->GetNeighbors(src);
      auto edges = store->GetOutEdges(src);
      CHECK_EQ(nbrs.Size(), edges.Size());
      for (int i = 0; i < nbrs.Size(); ++i) {
        LOG(INFO) << src << " -> " << nbrs[i] << ", edge_id = " << edges[i];
      }
    }
  }
  LOG(INFO) << "Passed graph-learn topo storage test...";

  LOG(INFO) << "Passed graph-learn fragment test...";

  grape::FinalizeMPIComm();

#endif

  return 0;
}

