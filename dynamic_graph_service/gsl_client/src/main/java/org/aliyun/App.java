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

        TFPredictClient client = new TFPredictClient(schema, "localhost", 9007);
        for (int i = 0; i < iters; ++i) {
          // Get Sampled EgoGraph from Service
          Value content = g.run(query);
          EgoGraph egoGraph = content.getEgoGraph();
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
