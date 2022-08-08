package org.aliyun;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;

import org.aliyun.gsl_client.DataSource;
import org.aliyun.gsl_client.Graph;
import org.aliyun.gsl_client.Query;
import org.aliyun.gsl_client.Value;
import org.aliyun.gsl_client.ValueBuilder;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.parser.PlanNode;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.predict.EgoGraph;
import org.aliyun.gsl_client.predict.TFPredictClient;
import org.aliyun.gsl_client.status.ErrorCode;
import org.aliyun.gsl_client.status.Status;


class MockDataSource extends DataSource {
  Vector<Long> vids = new Vector<>();
  int cursor = 0;
  int capacity = 0;

  MockDataSource(int capacity) {
    this.capacity = capacity;
    for (int i = 0; i < capacity; ++i) {
      vids.add(Long.valueOf(i));
    }
  }

  public Long next() {
    Long vid = vids.get(cursor);
    cursor += 1;
    return vid;
  }

  public boolean hasNext() {
    return cursor < capacity;
  }

  public boolean seekTimestamp(Long timestamp) {
    return true;
  }
}

class EgoGraphGenerator {
  private Query query;
  private Schema schema;

  EgoGraphGenerator(Query query, Schema schema) {
    this.query = query;
    this.schema = schema;
  }

  public EgoGraph mock() throws UserException {
    ValueBuilder builder = new ValueBuilder(query, 1 + 1 + 10 + 10 + 50, schema, 0L, 10);

    ArrayList<PlanNode> nodes = query.getPlan().getEgoGraphNodes();
    ArrayList<Integer> vtypes = new ArrayList<Integer>();
    ArrayList<Integer> etypes = new ArrayList<Integer>();
    ArrayList<Integer> vops = new ArrayList<Integer>();
    ArrayList<Integer> hops = new ArrayList<Integer>();
    ArrayList<Integer> eops = new ArrayList<Integer>();

    for (PlanNode node : nodes) {
      if (node.getKind().equals("EDGE_SAMPLER")) {
        eops.add(node.getId());
        hops.add(node.getParam("fanout"));
        etypes.add(node.getParam("etype"));
      } else if (node.getKind().equals("VERTEX_SAMPLER")) {
        vtypes.add(node.getParam("vtype"));
        vops.add(node.getId());
      } else {
        eops.add(node.getId());
        hops.add(1);
      }
    }

    builder.addVopRes(vops.get(0), (short)vtypes.get(0).intValue(), 0L, 1);
    builder.addEopRes(eops.get(1), (short)2, (short)0, (short)1, 0L, 10);
    for (int i = 0; i < 10; ++i) {
      builder.addVopRes(vops.get(1), (short)vtypes.get(1).intValue(), 0L + i, 1);
    }
    for (int i = 0; i < 10; ++i) {
      builder.addEopRes(eops.get(2), (short)3, (short)1, (short)1, 0L + i, 5);
    }
    for (int i = 0; i < 10 * 5; ++i) {
      builder.addVopRes(vops.get(2), (short)vtypes.get(2).intValue(), 0L + i, 1);
    }
    Value val = builder.finish();
    EgoGraph egoGraph = new EgoGraph(vtypes, vops, hops, eops);
    egoGraph = val.feedEgoGraph(egoGraph);
    return egoGraph;
  }
}

public class App
{
  public static void main( String[] args ) {
    String server = args[0];
    String modelName = args[1];
    Graph g = Graph.connect(server);


    int iters = 5;
    DataSource source = new MockDataSource(iters);

    try {
      Schema schema = g.getSchema();

      Query query = g.V("user").feed(source).properties(1).alias("seed")
          .outV("u2i").sample(10).by("topk_by_timestamp").properties(1).alias("hop1")
          .outV("i2i").sample(5).by("topk_by_timestamp").properties(1).alias("hop2")
          .values();
      Status s = g.install(query);

      // Uncomment me on other clients.
      // Query query = g.getQuery();
      // query.feed(source);
      // Status s = new Status(ErrorCode.OK);

      System.out.println("Query Installation Ready: " + query.getPlan().toJson().toString());

      if (s.ok()) {
        // Note: Make sure that DataLoader has set barrier("u2i_finished")
        while (!g.checkBarrier("u2i_finished").ok()) {
          Thread.sleep(1000);
          System.out.println("Barrier u2i_finished is not ready...");
        }

        TFPredictClient client = new TFPredictClient(schema, "localhost", 9000);
        for (int i = 0; i < iters; ++i) {
          // Get Sampled EgoGraph from Service
          Value content = g.run(query);
          EgoGraph egoGraph = content.getEgoGraph();

          // Uncomment me when mock sampled data.
          // EgoGraphGenerator gen = new EgoGraphGenerator(query, schema);
          // EgoGraph egoGraph = gen.mock();
          ArrayList<Integer> phs = new ArrayList<Integer>(Arrays.asList(0, 3, 4));
          client.predict(modelName, 1, egoGraph, phs);
        }
      } else {
        System.out.println("Install Query failed, error code:" + s.getCode());
      }
    } catch (Exception e) {
        e.printStackTrace();
    }
  }
}
