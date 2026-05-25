from __future__ import annotations


class RAGApplicationError(Exception):
    status_code = 500

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ConfigurationError(RAGApplicationError):
    status_code = 500


class DocumentParsingError(RAGApplicationError):
    status_code = 422


class EmbeddingError(RAGApplicationError):
    status_code = 502


class VectorStoreError(RAGApplicationError):
    status_code = 500


class RetrievalError(RAGApplicationError):
    status_code = 400


class LLMError(RAGApplicationError):
    status_code = 502


class AudioProcessingError(RAGApplicationError):
    status_code = 502


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    return chain


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = []
    for item in _exception_chain(exc):
        parts.append(f"{item.__class__.__name__}: {item}")
        for arg in getattr(item, "args", ()):
            if isinstance(arg, BaseException):
                parts.append(_exception_text(arg))
            elif arg:
                parts.append(str(arg))
    return " | ".join(part for part in parts if part).lower()


def is_ssl_error(exc: BaseException) -> bool:
    message = _exception_text(exc)
    return any(
        token in message
        for token in (
            "ssl",
            "tls",
            "certificate verify failed",
            "certifi",
            "cafile",
            "cert_verify_failed",
            "self-signed certificate",
            "unable to get local issuer certificate",
        )
    )


def clean_external_error(exc: BaseException, operation: str) -> str:
    if is_ssl_error(exc):
        return (
            f"{operation} failed because SSL/certificate verification could not be completed. "
            "The app now supports OPENAI_SSL_MODE=system to use the Windows/system trust store. "
            "If that still fails, set OPENAI_SSL_MODE=custom and OPENAI_CA_BUNDLE to your corporate CA bundle. "
            "As a last resort, use OPENAI_SSL_MODE=insecure only if your local policy allows it. "
            f"Details: {_short_error_detail(exc)}"
        )

    message_blob = _exception_text(exc)
    if any(token in message_blob for token in ("connection error", "connecterror", "timeout", "proxy")):
        return (
            f"{operation} failed because the OpenAI API connection could not be established. "
            "Check VPN/proxy/firewall settings and SSL mode. "
            f"Details: {_short_error_detail(exc)}"
        )

    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"{operation} failed: {message}"


def _short_error_detail(exc: BaseException, max_length: int = 600) -> str:
    details = " | ".join(
        f"{item.__class__.__name__}: {str(item).strip() or repr(item)}"
        for item in _exception_chain(exc)
    )
    details = details.strip() or exc.__class__.__name__
    if len(details) <= max_length:
        return details
    return details[: max_length - 3] + "..."
