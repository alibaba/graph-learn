package org.aliyun.gsl_client;

import java.util.concurrent.CompletableFuture;

import org.aliyun.gsl_client.exception.UserException;
import org.aliyun.gsl_client.http.HttpConfig;
import org.aliyun.gsl_client.impl.GraphImpl;
import org.aliyun.gsl_client.parser.schema.Schema;
import org.aliyun.gsl_client.status.Status;


public interface Graph {
  /**
   * Connect to Remote Graph Service with serverAddr.
   * @param server_addr
   * @return Graph
   */
  static Graph connect(String serverAddr) {
    return new GraphImpl(serverAddr);
  }

  /**
   * Connect to Remote Graph Service with user configured HttpConfig.
   * This is an advance usage for performance.
   * @param config
   * @return Graph
   */
  static Graph connect(HttpConfig config) {
    return new GraphImpl(config);
  }

  /**
   * Get the graph schema.
   * @return Schema object
   * @throws UserException
   */
  Schema getSchema() throws UserException;

  /**
   * Start the GSL traversal on Graph.
   * @param vtype
   * @return Traversal object
   * @throws UserException
   */
  Traversal V(String vtype) throws UserException;

  /**
   * Async Install Query which is generated from traversal on Graph and
   * ends up with `values()`.
   * @param query
   * @return CompletableFuture<Status>, Status is OK when query is installed
   * sucessfully, otherwise return ERROR Status immediately.
   */
  CompletableFuture<Status> installAsync(Query query);

  /**
   * Install Query with is generated from traversal on Graph and ends
   * up with `values()`.
   * @param query
   * @return Status, wait for query installation in another thread and then return Status OK.
   * Otherwise, return ERROR Status immediately.
   */
  Status install(Query query);

  Value run(Query query) throws UserException;
}
