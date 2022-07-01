package org.aliyun.gsl_client.http;

import java.io.IOException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;

import java.util.Arrays;
import java.util.List;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.atomic.AtomicLong;
import org.json.JSONObject;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Proxy;

import okhttp3.Handshake;
import okhttp3.HttpUrl;

import okhttp3.Dispatcher;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;
import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.ConnectionPool;
import okhttp3.Protocol;
import okhttp3.Connection;
import okhttp3.EventListener;
import okhttp3.internal.Util;
import okhttp3.ConnectionSpec;

/**
 * Event listener for minitoring http client behaviors
 */
public class PrintingEventListener extends EventListener {
  private static final Factory FACTORY = new Factory() {
    final AtomicLong nextCallId = new AtomicLong(1L);

    @Override public EventListener create(Call call) {
      long callId = nextCallId.getAndIncrement();
      System.out.printf("%04d %s%n", callId, call.request().url());
      return new PrintingEventListener(callId, System.nanoTime());
    }
  };

  final long callId;
  final long callStartNanos;

  PrintingEventListener(long callId, long callStartNanos) {
    this.callId = callId;
    this.callStartNanos = callStartNanos;
  }

  private void printEvent(String name) {
    long elapsedNanos = System.nanoTime() - callStartNanos;
    System.out.printf("%04d %.3f %s%n", callId, elapsedNanos / 1000000000d, name);
  }

  @Override public void proxySelectStart(Call call, HttpUrl url) {
    printEvent("proxySelectStart");
  }

  @Override public void proxySelectEnd(Call call, HttpUrl url, List<Proxy> proxies) {
    printEvent("proxySelectEnd");
  }

  @Override public void callStart(Call call) {
    printEvent("callStart");
  }

  @Override public void dnsStart(Call call, String domainName) {
    printEvent("dnsStart");
  }

  @Override public void dnsEnd(Call call, String domainName, List<InetAddress> inetAddressList) {
    printEvent("dnsEnd");
  }

  @Override public void connectStart(
      Call call, InetSocketAddress inetSocketAddress, Proxy proxy) {
    printEvent("connectStart");
  }

  @Override public void secureConnectStart(Call call) {
    printEvent("secureConnectStart");
  }

  @Override public void secureConnectEnd(Call call, Handshake handshake) {
    printEvent("secureConnectEnd");
  }

  @Override public void connectEnd(
      Call call, InetSocketAddress inetSocketAddress, Proxy proxy, Protocol protocol) {
    printEvent("connectEnd");
  }

  @Override public void connectFailed(Call call, InetSocketAddress inetSocketAddress, Proxy proxy,
      Protocol protocol, IOException ioe) {
    printEvent("connectFailed");
  }

  @Override public void connectionAcquired(Call call, Connection connection) {
    printEvent("connectionAcquired");
  }

  @Override public void connectionReleased(Call call, Connection connection) {
    printEvent("connectionReleased");
  }

  @Override public void requestHeadersStart(Call call) {
    printEvent("requestHeadersStart");
  }

  @Override public void requestHeadersEnd(Call call, Request request) {
    printEvent("requestHeadersEnd");
  }

  @Override public void requestBodyStart(Call call) {
    printEvent("requestBodyStart");
  }

  @Override public void requestBodyEnd(Call call, long byteCount) {
    printEvent("requestBodyEnd");
  }

  @Override public void requestFailed(Call call, IOException ioe) {
    printEvent("requestFailed");
  }

  @Override public void responseHeadersStart(Call call) {
    printEvent("responseHeadersStart");
  }

  @Override public void responseHeadersEnd(Call call, Response response) {
    printEvent("responseHeadersEnd");
  }

  @Override public void responseBodyStart(Call call) {
    printEvent("responseBodyStart");
  }

  @Override public void responseBodyEnd(Call call, long byteCount) {
    printEvent("responseBodyEnd");
  }

  @Override public void responseFailed(Call call, IOException ioe) {
    printEvent("responseFailed");
  }

  @Override public void callEnd(Call call) {
    printEvent("callEnd");
  }

  @Override public void callFailed(Call call, IOException ioe) {
    printEvent("callFailed");
  }

  @Override public void canceled(Call call) {
    printEvent("canceled");
  }
}