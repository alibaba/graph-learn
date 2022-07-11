package org.aliyun.gsl_client.parser.optimizer;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

import org.aliyun.gsl_client.parser.PlanNode;

public class FusionRule extends Rule{
  // vertex type to index.
  private Map<Integer, Integer> vtype2Idx = new HashMap<>();

  public void Act(Vector<PlanNode> nodes, AtomicInteger size) {
    int i = 0;
    int j = size.get() - 1;

    for (; i <= j; ++i) {
      while (i <= j && FuseAndSwap(nodes, i, j)) {
        j -= 1;
      }
    }
    size.set(j + 1);;
    for (int idx = 0; idx <= j; ++idx) {
      nodes.get(idx).setId(idx);
    }
  }

  // Fuse same VertexSamplerNode, and swap out the redundant PlanNode
  private boolean FuseAndSwap(Vector<PlanNode> nodes, int i, int j) {
    PlanNode node = nodes.get(i);
    if (node.getKind().equals("VERTEX_SAMPLER")) {
      Integer vtype = node.getParam("vtype");
      if (vtype2Idx.containsKey(vtype)) {
        PlanNode pre_node = nodes.get(vtype2Idx.get(vtype));
        assert(node.getChildLinks().isEmpty());
        Integer version = Math.max(pre_node.getParam("versions"), node.getParam("versions"));
        pre_node.addParam("versions", version);
        node.copyFrom(pre_node);
        Collections.swap(nodes, i, j);
        return true;
      } else {
        vtype2Idx.put(vtype, node.getId());
      }
    }
    return false;
  }
}
