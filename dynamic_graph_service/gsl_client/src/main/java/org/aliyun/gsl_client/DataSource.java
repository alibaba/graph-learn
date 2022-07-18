package org.aliyun.gsl_client;

/**
 * This is the base DataSouce, user-defined DataSource must implement
 * next(), hashNext(), and seekTimestamp().
 */
public abstract class DataSource {

  /**
   * Get the next request vertex id from data source, most streaming data source
   * may wait util there are next data.
   * @return Long, vertex id for query input.
   */
  public abstract Long next();

  /**
   * If the data source has next data or not.
   * @return boolean, return true when DataSource is active and has next data,
   * otherwise return false.
   */
  public abstract boolean hasNext();

  /**
   * Seek for the position of given timestamp, and moving begining of data
   * source to that position.
   * @return boolean, return true when seek and move succeed, otherwise return
   * false.
   */
  public abstract boolean seekTimestamp(Long timestamp);

}
