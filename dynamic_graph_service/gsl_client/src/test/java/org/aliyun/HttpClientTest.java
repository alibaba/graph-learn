package org.aliyun;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CompletableFuture;

import org.aliyun.gsl_client.Graph;
import org.aliyun.gsl_client.Query;
import org.aliyun.gsl_client.Value;
import org.aliyun.gsl_client.http.HttpClient;
import org.aliyun.gsl_client.http.HttpConfig;
import org.aliyun.gsl_client.parser.schema.Schema;

/**
 * Run these test cases after `python3 mock_server.py`.
 */
public class HttpClientTest extends TestCase {
  private HttpClient client;
  private HttpConfig config;

  public HttpClientTest(String testName) {
    super(testName);
    String mockServer = "http://127.0.0.1:8088";
    config = new HttpConfig();
    config.setServerAddr(mockServer);
    client = new HttpClient(config);
  }

  public static Test suite() {
    return new TestSuite(HttpClientTest.class);
  }

  public void testInstall() {
    Graph g = Graph.connect(config);
    try {
      Query query = g.V("user").properties(10)
          .outV("knows").sample(15).by("topk").properties(10)
          .outV("knows").sample(10).by("topk").properties(10)
          .values();
      CompletableFuture<byte[]> fut = client.install(query.getPlan().toJson());
      byte[] content = fut.join();
      String s = new String(content, StandardCharsets.UTF_8);
      assertEquals(s, "0");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void testRun() {
    try {
      String queryId = "0";
      long input = 2;
      CompletableFuture<byte[]> fut = client.run(queryId, input);
      byte[] content = fut.join();
      String s = new String(content, StandardCharsets.UTF_8);
      assertEquals(s, "HTTP RESPONSE: Run Query SUCCEED.\n");
      // In e2e test, we should check the flatbuffers response value.
      // Value value = new Value(content);
      // assertTrue(value.vaild());
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void testGetSchema() {
    try {
      CompletableFuture<byte[]> fut = client.getSchema();
      byte[] content = fut.join();
      Schema schema =  Schema.parseFrom(content);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
