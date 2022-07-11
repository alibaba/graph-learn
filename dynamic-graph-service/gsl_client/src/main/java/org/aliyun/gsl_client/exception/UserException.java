package org.aliyun.gsl_client.exception;

import org.aliyun.gsl_client.status.ErrorCode;

public class UserException extends Exception {
  private int code;
  private String message;

  public UserException() {
    super();
  }

  public UserException(Exception e) {
    super(e);
  }

  public UserException(ErrorCode code, String message) {
    super(message);
    this.code = code.getValue();
  }

  public UserException(ErrorCode code, String message, Throwable cause) {
      super(message, cause);
      this.code = code.getValue();
  }

  public int getCode() {
      return code;
  }

  public String getMessage() {
    return message;
  }
}
