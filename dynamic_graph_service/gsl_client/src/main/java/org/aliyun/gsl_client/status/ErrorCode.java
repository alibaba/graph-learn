package org.aliyun.gsl_client.status;


public enum ErrorCode {
  UNKNOWN       (-1, false),
  OK             (0, true),
  HTTP_ERROR     (1, false),
  PAESE_ERROR    (2, false),
  INTERNAL_ERROR (3, false);

  private final int value;
  private final boolean ok;

  private ErrorCode(int value, boolean ok) {
    this.value = value;
    this.ok = ok;
  }

  public boolean ok() {
    return ok;
  }

  public int getValue() {
    return value;
  }
}
