package org.aliyun.gsl_client.impl;

import org.aliyun.gsl_client.parser.Context;
import org.aliyun.gsl_client.parser.Plan;
import org.aliyun.gsl_client.parser.PlanNode;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.status.ErrorCode;
import org.aliyun.gsl_client.status.Status;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.json.JSONObject;

import java.nio.charset.StandardCharsets;
import java.util.concurrent.CompletableFuture;

import org.aliyun.gsl_client.Graph;
import org.aliyun.gsl_client.Query;
import org.aliyun.gsl_client.Traversal;
import org.aliyun.gsl_client.Value;
import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.http.HttpClient;
import org.aliyun.gsl_client.http.HttpConfig;

public class GraphImpl implements Graph {
  private Traversal traversal;
  private HttpClient client;
  private Schema schema;
  private static Log log = LogFactory.getLog(GraphImpl.class);

  public GraphImpl(String serverAddr){
    this.traversal = new Traversal();
    HttpConfig config = new HttpConfig();
    config.setServerAddr(serverAddr);
    this.client = new HttpClient(config);
    try {
      getSchema();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public GraphImpl(HttpConfig config){
    this.traversal = new Traversal();
    this.client = new HttpClient(config);
    try {
      getSchema();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public Traversal V(String vtype) throws UserException {
    Plan plan = new Plan();
    Context ctx = new Context(plan, this, 0, vtype, this.schema);

    try {
      PlanNode root = plan.addRoot(ctx.type(vtype));
      this.traversal.context = ctx;
      this.traversal.plan_node = root;
      return this.traversal;
    } catch (Exception e) {
      throw new UserException(ErrorCode.PARSE_ERROR, e.getMessage());
    }
  }

  public CompletableFuture<Status> installAsync(Query query) {
    JSONObject queryPlan = null;
    CompletableFuture<Status> fut = new CompletableFuture<>();
    try {
      queryPlan = query.getPlan().toJson();
      fut = client.install(queryPlan).thenApply(res -> {
        if (res != null) {
          String qid = res.toString();
          query.setId(qid);
          return new Status(ErrorCode.OK);
        } else {
          query.setId("-1");
          return new Status(ErrorCode.HTTP_ERROR);
        }
      });
    } catch (UserException e) {
      e.printStackTrace();
      fut.completeExceptionally(e);
    }
    return fut;
  }

  public Status install(Query query) {
    JSONObject queryPlan = null;
    try {
      queryPlan = query.getPlan().toJson();
      CompletableFuture<byte[]> fut = client.install(queryPlan);
      byte[] res = fut.join();
      if (res != null) {
        String qid = new String(res, StandardCharsets.UTF_8);
        query.setId(qid);
        return new Status(ErrorCode.OK);
      } else {
        query.setId("-1");
        return new Status(ErrorCode.PARSE_ERROR);
      }
    } catch (UserException e) {
      e.printStackTrace();
      return new Status(ErrorCode.HTTP_ERROR);
    }
  }

  public CompletableFuture<Value> runAsync(Query query) throws UserException {
    CompletableFuture<Value> fut = new CompletableFuture<>();
    try {
      long input = query.getPlan().root().getSource().next();
      fut = client.run(query.getId(), input).thenApply(res -> {
        try {
         Value value = new Value(query, input, res);
         return value;
        } catch (UserException e) {
          e.printStackTrace();
          return null;
        }
      });
    } catch (UserException e) {
      e.printStackTrace();
      fut.completeExceptionally(e);
    }
    return fut;
  }

  public Value run(Query query) throws UserException {
    byte[] content = null;
    Long input = null;
    try {
      input = query.getPlan().root().getSource().next();
      CompletableFuture<byte[]> fut = client.run(query.getId(), input);
      content = fut.join();
      Value value = new Value(query, input, content);
      return value;
    } catch (Exception e) {
      throw new UserException(ErrorCode.HTTP_ERROR, e.getMessage());
    }
  }

  public Schema getSchema() throws UserException {
    if (this.schema != null) {
      return this.schema;
    }
    byte[] content = null;
    try {
      CompletableFuture<byte[]> fut = client.getSchema();
      content = fut.join();
    } catch (Exception e) {
      e.printStackTrace();
      throw new UserException(ErrorCode.HTTP_ERROR, e.getMessage());
    }
    try {
      this.schema = Schema.parseFrom(content);
    } catch (Exception e) {
      e.printStackTrace();
      throw new UserException(ErrorCode.PARSE_ERROR, e.getMessage());
    }
    log.info("Successfully get schema from server :)");
    return this.schema;
  }

  public Status checkBarrier(String name) throws UserException {
    byte[] content = null;
    try {
      CompletableFuture<byte[]> fut = client.checkBarrier(name);
      content = fut.join();
    } catch (Exception e) {
      e.printStackTrace();
      throw new UserException(ErrorCode.HTTP_ERROR, e.getMessage());
    }
    String res = new String(content);
    if (res.equals("READY")) {
      return new Status(ErrorCode.OK);
    } else {
      System.out.println("CheckBarrier with status " + res);
      return new Status(ErrorCode.NOT_READY_ERROR);
    }
  }

  public Query getQuery() throws UserException {
    byte[] content = null;
    try {
      CompletableFuture<byte[]> fut = client.getQuery();
      content = fut.join();
      Plan plan = Plan.parseFrom(content);
      Query query = new Query(plan);
      query.setId("0");
      return query;
    } catch (Exception e) {
      e.printStackTrace();
      throw new UserException(ErrorCode.HTTP_ERROR, e.getMessage());
    }
  }
}
