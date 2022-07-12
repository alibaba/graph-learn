package org.aliyun.gsl_client.http;

import java.io.IOException;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletableFuture;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.json.JSONObject;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.ConnectionPool;
import okhttp3.Dispatcher;
import okhttp3.OkHttpClient;
import okhttp3.internal.Util;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;

class ResponseFuture implements Callback {
  final CompletableFuture<Response> future = new CompletableFuture<>();

  public ResponseFuture() {
  }

  @Override
  public void onFailure(Call call, IOException e) {
    e.printStackTrace();
    future.completeExceptionally(e);
  }

  @Override
  public void onResponse(Call call, Response response) throws IOException {
    future.complete(response);
  }
}

public class HttpClient {
  private OkHttpClient client = null;
  private String uri;
  private String contentType = "text/plain";
  private static Log log = LogFactory.getLog(HttpClient.class);

  public HttpClient(HttpConfig config) {
    client = new OkHttpClient.Builder()
      .dispatcher(new Dispatcher(new ThreadPoolExecutor(0, Integer.MAX_VALUE,
                                  60L, TimeUnit.SECONDS,
                                  new SynchronousQueue<>(),
                                  Util.threadFactory("OkHttp Dispatcher", false)
                                  )))
      .connectionPool(new ConnectionPool(config.getConnectPoolMaxIdle(),
                                         config.getKeepAliveDuration(),
                                         TimeUnit.MILLISECONDS))
      .readTimeout(config.getReadTimeout(), TimeUnit.MILLISECONDS)
      .connectTimeout(config.getConnectTimeout(), TimeUnit.MILLISECONDS)
      .build();
    client.dispatcher().setMaxRequests(config.getMaxRequests());
    client.dispatcher().setMaxRequestsPerHost(config.getMaxRequestsPerHost());
    this.uri = config.getServerAddr();
  }

  public CompletableFuture<byte[]> install(JSONObject queryPlan) {
    String planStr = queryPlan.toString();
    return install(planStr);
  }

  public CompletableFuture<byte[]> install(String queryPlan){
    log.info("Install query:" + queryPlan);
    RequestBody body = RequestBody.create(queryPlan.getBytes());

    Request request = new Request.Builder()
            .url(uri + "/admin/init/")
            .post(body)
            .header("Connection", "Keep-Alive")
            .header("Content-Type", contentType)
            .build();

    CompletableFuture<byte[]> content = getContent(request);
    return content;
  }

  public CompletableFuture<byte[]> run(String queryId, Long input) {
    StringBuilder url = new StringBuilder(64);
    url.append(uri);
    url.append("/infer?");
    url.append("qid=");
    url.append(queryId);
    url.append("&vid=");
    url.append(input);
    Request request = new Request.Builder()
            .url(url.toString())
            .header("Connection", "Keep-Alive")
            .header("Content-Type", contentType)
            .build();

    CompletableFuture<byte[]> content = getContent(request);
    return content;
  }

  public CompletableFuture<byte[]> getSchema() {
    Request request = new Request.Builder()
            .url(uri + "/admin/schema/")
            .header("Connection", "Keep-Alive")
            .header("Content-Type", contentType)
            .build();

    CompletableFuture<byte[]> content = getContent(request);
    return content;
  }

  CompletableFuture<byte[]> getContent(Request request) {
    ResponseFuture callback = new ResponseFuture();
    client.newCall(request).enqueue(callback);

    return callback.future.thenApply(response -> {
      try {
        ResponseBody responseBody = response.body();
        return responseBody.bytes();
      } catch (IOException e) {
        e.printStackTrace();
        return null;
      } finally {
        response.close();
      }
    });
  }
}
