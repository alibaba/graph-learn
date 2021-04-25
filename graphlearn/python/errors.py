# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from graphlearn import pywrap_graphlearn as pywrap

class BaseError(Exception):
  """A generic error that is raised when GraphLearn execution fails.
  Whenever possible, it will raise a more specific subclass of `BaseError`.
  """

  def __init__(self, message, error_code):
    """Creates a new `BaseError` indicating a particular fail.

    Args:
      message: The message string describing the failure.
      error_code: The `pywrap.ErrorCode.Code` describing the error.
    """
    super(BaseError, self).__init__()
    self._message = message
    self._error_code = error_code

  @property
  def message(self):
    """The error message that describes the error."""
    return self._message

  @property
  def error_code(self):
    """The integer error code that describes the error."""
    return self._error_code

  def __str__(self):
    return self.message


OK = pywrap.ErrorCode.OK
CANCELLED = pywrap.ErrorCode.CANCELLED
UNKNOWN = pywrap.ErrorCode.UNKNOWN
INVALID_ARGUMENT = pywrap.ErrorCode.INVALID_ARGUMENT
DEADLINE_EXCEEDED = pywrap.ErrorCode.DEADLINE_EXCEEDED
NOT_FOUND = pywrap.ErrorCode.NOT_FOUND
ALREADY_EXISTS = pywrap.ErrorCode.ALREADY_EXISTS
PERMISSION_DENIED = pywrap.ErrorCode.PERMISSION_DENIED
UNAUTHENTICATED = pywrap.ErrorCode.UNAUTHENTICATED
RESOURCE_EXHAUSTED = pywrap.ErrorCode.RESOURCE_EXHAUSTED
FAILED_PRECONDITION = pywrap.ErrorCode.FAILED_PRECONDITION
ABORTED = pywrap.ErrorCode.ABORTED
OUT_OF_RANGE = pywrap.ErrorCode.OUT_OF_RANGE
UNIMPLEMENTED = pywrap.ErrorCode.UNIMPLEMENTED
INTERNAL = pywrap.ErrorCode.INTERNAL
UNAVAILABLE = pywrap.ErrorCode.UNAVAILABLE
DATA_LOSS = pywrap.ErrorCode.DATA_LOSS
REQUEST_STOP = pywrap.ErrorCode.REQUEST_STOP


class CancelledError(BaseError):
  def __init__(self, message):
    """Creates a `CancelledError`."""
    super(CancelledError, self).__init__(message, CANCELLED)


class UnknownError(BaseError):
  def __init__(self, message):
    """Creates an `UnknownError`."""
    super(UnknownError, self).__init__(message, UNKNOWN)


class InvalidArgumentError(BaseError):
  def __init__(self, message):
    """Creates an `InvalidArgumentError`."""
    super(InvalidArgumentError, self).__init__(message, INVALID_ARGUMENT)


class DeadlineExceededError(BaseError):
  def __init__(self, message):
    """Creates a `DeadlineExceededError`."""
    super(DeadlineExceededError, self).__init__(message, DEADLINE_EXCEEDED)


class NotFoundError(BaseError):
  def __init__(self, message):
    """Creates a `NotFoundError`."""
    super(NotFoundError, self).__init__(message, NOT_FOUND)


class AlreadyExistsError(BaseError):
  def __init__(self, message):
    """Creates an `AlreadyExistsError`."""
    super(AlreadyExistsError, self).__init__(message, ALREADY_EXISTS)


class PermissionDeniedError(BaseError):
  def __init__(self, message):
    """Creates a `PermissionDeniedError`."""
    super(PermissionDeniedError, self).__init__(message, PERMISSION_DENIED)


class UnauthenticatedError(BaseError):
  def __init__(self, message):
    """Creates an `UnauthenticatedError`."""
    super(UnauthenticatedError, self).__init__(message, UNAUTHENTICATED)


class ResourceExhaustedError(BaseError):
  def __init__(self, message):
    """Creates a `ResourceExhaustedError`."""
    super(ResourceExhaustedError, self).__init__(message, RESOURCE_EXHAUSTED)


class FailedPreconditionError(BaseError):
  def __init__(self, message):
    """Creates a `FailedPreconditionError`."""
    super(FailedPreconditionError, self).__init__(message, FAILED_PRECONDITION)


class AbortedError(BaseError):
  def __init__(self, message):
    """Creates an `AbortedError`."""
    super(AbortedError, self).__init__(message, ABORTED)


class OutOfRangeError(BaseError):
  def __init__(self, message):
    """Creates an `OutOfRangeError`."""
    super(OutOfRangeError, self).__init__(message, OUT_OF_RANGE)


class UnimplementedError(BaseError):
  def __init__(self, message):
    """Creates an `UnimplementedError`."""
    super(UnimplementedError, self).__init__(message, UNIMPLEMENTED)


class InternalError(BaseError):
  def __init__(self, message):
    """Creates an `InternalError`."""
    super(InternalError, self).__init__(message, INTERNAL)


class UnavailableError(BaseError):
  def __init__(self, message):
    """Creates an `UnavailableError`."""
    super(UnavailableError, self).__init__(message, UNAVAILABLE)


class DataLossError(BaseError):
  def __init__(self, message):
    """Creates a `DataLossError`."""
    super(DataLossError, self).__init__(message, DATA_LOSS)


class RequestStopError(BaseError):
  def __init__(self, message):
    """Create a `RequestStopError`."""
    super(RequestStopError, self).__init__(message, REQUEST_STOP)


_CODE_TO_EXCEPTION_CLASS = {
    CANCELLED: CancelledError,
    UNKNOWN: UnknownError,
    INVALID_ARGUMENT: InvalidArgumentError,
    DEADLINE_EXCEEDED: DeadlineExceededError,
    NOT_FOUND: NotFoundError,
    ALREADY_EXISTS: AlreadyExistsError,
    PERMISSION_DENIED: PermissionDeniedError,
    UNAUTHENTICATED: UnauthenticatedError,
    RESOURCE_EXHAUSTED: ResourceExhaustedError,
    FAILED_PRECONDITION: FailedPreconditionError,
    ABORTED: AbortedError,
    OUT_OF_RANGE: OutOfRangeError,
    UNIMPLEMENTED: UnimplementedError,
    INTERNAL: InternalError,
    UNAVAILABLE: UnavailableError,
    DATA_LOSS: DataLossError,
    REQUEST_STOP: RequestStopError,
}

_EXCEPTION_CLASS_TO_CODE = dict((
    (class_, code) for (code, class_) in _CODE_TO_EXCEPTION_CLASS.items()))


def exception_type_from_error_code(error_code):
  return _CODE_TO_EXCEPTION_CLASS[error_code]


def error_code_from_exception_type(cls):
  return _EXCEPTION_CLASS_TO_CODE[cls]


def _make_specific_exception(message, error_code):
  try:
    exc_type = exception_type_from_error_code(error_code)
    return exc_type(message)
  except KeyError:
    warnings.warn("Unknown error code: %d" % error_code)
    return UnknownError(message)


def raise_exception_on_not_ok_status(status):
  if not status.ok():
    raise _make_specific_exception(status.message(), int(status.code()))
