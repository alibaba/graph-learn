package org.aliyun;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;

import org.aliyun.gsl_client.DataSource;
import org.aliyun.gsl_client.Graph;
import org.aliyun.gsl_client.Query;
import org.aliyun.gsl_client.Value;
import org.aliyun.gsl_client.predict.EgoGraph;
import org.aliyun.gsl_client.predict.TFPredictClient;
import org.aliyun.gsl_client.status.Status;
import org.aliyun.gsl_client.Decoder;


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
    String server = "http://dynamic-graph-service.info";
    Graph g = Graph.connect(server);

    int iters = 5;
    DataSource source = new MockDataSource(iters);
    Instant start = Instant.now();
    try {
        Query query = g.V("user").feed(source).properties(1).alias("seed")
            .outV("u2i").sample(15).by("topk_by_timestamp").properties(1).alias("hop1")
            .outV("i2i").sample(10).by("topk_by_timestamp").properties(1).alias("hop2")
            .values();
        System.out.println("Waiting for Query Installation Ready...");
        Status s = g.install(query);
        if (s.ok()) {
          // describe the attributes with alias in query.
          Decoder decoder = new Decoder(g);
          decoder.addFeatDesc("user",
                              new ArrayList<String>(Arrays.asList("string", "float")),
                              new ArrayList<Integer>(Arrays.asList(1, 100)));
          decoder.addFeatDesc("item",
                              new ArrayList<String>(Arrays.asList("string", "float")),
                              new ArrayList<Integer>(Arrays.asList(1, 100)));

          System.out.println("Install Query succeed, query id: " + query.getId());
          TFPredictClient client = new TFPredictClient(decoder, "localhost", 9000);
          for (int i = 0; i < iters; ++i) {
            Value content = g.run(query);
            EgoGraph egoGraph = content.getEgoGraph("seed");
            client.predict("model", 1, egoGraph);
          }
        } else {
          System.out.println("Install Query failed, error code:" + s.getCode());
        }
      } catch (Exception e) {
          e.printStackTrace();
      }
      // for other clients
      // DataSource source = ..;
      // Graph g = Graph.connect(server);
      // Query query = g.getQuery(queryId);
      // query.feed("seed", source);
      // Value val = q.run(query)

      // desc attr decoder

      Instant end = Instant.now();
      Duration timeElapsed = Duration.between(start, end);
      System.out.println("AppTest Time taken: "+ timeElapsed.toMillis() +" milliseconds");
    }
}
