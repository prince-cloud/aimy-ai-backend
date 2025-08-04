from rest_framework.exceptions import APIException, ValidationError
from rest_framework.response import Response
from rest_framework.views import exception_handler

import logging

logger = logging.getLogger(__name__)


class BaseException(APIException):
    status_code = 400

    def __init__(
        self,
        detail=None,
        response_code=None,
        error_code=None,
    ):
        self.response_code = response_code
        self.error_code = error_code
        self.detail = detail or self.default_detail


class GenericFieldException(BaseException):
    status_code = 400
    default_code = 400
    default_detail = (
        "A field exception occured. Find the exact fields as part of this json data."
    )


class ServerFaultException(BaseException):
    status_code = 500
    default_code = 500
    default_detail = "An error occured on the server. Please report to the admin."


def custom_exception_handler(exception: Exception, context):
    status_code = 500
    code = 500
    details = "Some unexpected exception occured"

    # Example of how these could be branched out further with error message and status code edited
    if isinstance(exception, BaseException):
        status_code = exception.response_code or exception.status_code
        code = exception.error_code or exception.default_code
        details = exception.detail or exception.default_detail

    else:
        response = exception_handler(exception, context)
        if isinstance(exception, ValidationError):
            code = GenericFieldException.default_code
            details = GenericFieldException.default_detail
        elif response is None or response.status_code == 500:
            code = ServerFaultException.default_code
            details = ServerFaultException.default_detail
            logger.exception(exception)
        if response:
            response.data["errorCode"] = code
            response.data["errorMsg"] = details
            return response

    data = {
        "status_code": status_code,
        "errorCode": code,
        "errorMsg": details,
    }
    return Response(data, status=status_code)
