"""
MCP Authorization support for mcp-server-lammps.

Implements OAuth 2.0 Bearer token verification per MCP specification (2025-11-25):
- Bearer token validation (OAuth 2.1 Section 5)
- Protected Resource Metadata (RFC 9728)
- WWW-Authenticate headers with resource_metadata and scope
- IP/CIDR allowlist for trusted network deployments

This module provides three auth modes:
1. Full OAuth Bearer auth (default for `serve` mode)
2. --skip-auth: No authentication required
3. --allowed-ips: IP/CIDR whitelist bypasses token check; others require Bearer token
"""

from __future__ import annotations

import ipaddress
import json
import logging
import secrets
import time
from ipaddress import IPv4Network, IPv6Network
from typing import Sequence

from starlette.types import ASGIApp, Receive, Scope, Send

from mcp.server.auth.provider import AccessToken, TokenVerifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token generation & verification for self-hosted (standalone) mode
# ---------------------------------------------------------------------------


class StaticTokenVerifier:
    """
    Simple token verifier that checks against a single server-generated token.

    Used when the MCP server acts as its own authorization server (standalone
    deployment). A random token is generated at startup and printed to stderr
    so the operator can distribute it to authorized clients.

    Tokens are Bearer tokens per OAuth 2.1 §5.1.1.
    """

    def __init__(self, token: str | None = None, expires_in: int | None = None) -> None:
        self.token = token or secrets.token_urlsafe(32)
        self.created_at = int(time.time())
        self.expires_at = (self.created_at + expires_in) if expires_in else None
        self._revoked = False

    async def verify_token(self, token: str) -> AccessToken | None:
        if self._revoked:
            return None
        if token != self.token:
            return None
        if self.expires_at and int(time.time()) > self.expires_at:
            return None
        return AccessToken(
            token=token,
            client_id="static",
            scopes=["mcp:full"],
            expires_at=self.expires_at,
        )

    def revoke(self) -> None:
        self._revoked = True


# ---------------------------------------------------------------------------
# IP allowlist middleware
# ---------------------------------------------------------------------------

IpNetwork = IPv4Network | IPv6Network


def parse_ip_allowlist(raw: Sequence[str]) -> list[IpNetwork]:
    """Parse a list of IP addresses or CIDR notations into network objects.

    Accepts:
      - Single IPs: "192.168.1.1", "::1"
      - CIDR ranges: "192.168.0.0/16", "10.0.0.0/8", "fd00::/8"
      - Always includes loopback (127.0.0.1/8, ::1/128) implicitly.
    """
    networks: list[IpNetwork] = []
    for entry in raw:
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError as exc:
            raise ValueError(f"Invalid IP/CIDR in allowlist: {entry!r}") from exc
    # Always include loopback
    networks.append(IPv4Network("127.0.0.0/8"))
    networks.append(IPv6Network("::1/128"))
    return networks


def _get_client_ip(scope: Scope) -> str | None:
    """Extract client IP from ASGI scope, respecting X-Forwarded-For only if
    the immediate peer is a trusted proxy (loopback)."""
    client = scope.get("client")
    if client is None:
        return None
    peer_ip = client[0]

    # Only trust X-Forwarded-For from loopback (reverse proxy on same host)
    try:
        peer = ipaddress.ip_address(peer_ip)
        if peer.is_loopback:
            headers = dict(scope.get("headers", []))
            xff = headers.get(b"x-forwarded-for", b"").decode()
            if xff:
                return xff.split(",")[0].strip()
    except ValueError:
        pass
    return peer_ip


def ip_in_allowlist(ip_str: str, allowlist: list[IpNetwork]) -> bool:
    """Check if an IP address falls within any of the allowed networks."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(addr in net for net in allowlist)


# WebSocket close code for auth/policy failures (RFC 6455 §7.4.1)
WS_POLICY_VIOLATION = 1008


async def _deny_websocket(send: Send, code: int = WS_POLICY_VIOLATION) -> None:
    """Reject a websocket connection before accept with a close frame.

    Per ASGI spec, sending ``websocket.close`` during the CONNECTING state
    causes the server (uvicorn) to reject the HTTP upgrade with 403.
    """
    await send({"type": "websocket.close", "code": code})


class IPAllowlistMiddleware:
    """
    ASGI middleware that allows requests from whitelisted IPs without auth.

    Requests from non-whitelisted IPs are rejected with 403 Forbidden.
    When combined with --skip-auth, this provides zero-auth for trusted IPs.
    When combined with Bearer auth, whitelisted IPs bypass token verification
    while others must provide a valid Bearer token.
    """

    def __init__(
        self,
        app: ASGIApp,
        allowlist: list[IpNetwork],
        *,
        reject_non_listed: bool = True,
    ) -> None:
        self.app = app
        self.allowlist = allowlist
        self.reject_non_listed = reject_non_listed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        client_ip = _get_client_ip(scope)
        if client_ip and ip_in_allowlist(client_ip, self.allowlist):
            # Whitelisted — pass through without auth
            await self.app(scope, receive, send)
            return

        if self.reject_non_listed:
            if scope["type"] == "websocket":
                await _deny_websocket(send)
                return
            body = json.dumps(
                {
                    "error": "forbidden",
                    "error_description": f"Client IP {client_ip} is not in the allowlist.",
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        # Not in allowlist but not rejecting — pass through (Bearer auth will handle)
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Combined auth middleware: IP allowlist + Bearer token
# ---------------------------------------------------------------------------


class CombinedAuthMiddleware:
    """
    ASGI middleware that checks IP allowlist first, then falls back to
    Bearer token verification.

    - If client IP is in the allowlist → pass through (no token needed)
    - If client IP is NOT in the allowlist → require valid Bearer token
    - If --skip-auth is enabled → this middleware is not used at all

    Emits proper WWW-Authenticate headers per MCP spec:
    - 401 with resource_metadata URL for missing/invalid tokens
    - 403 with error=insufficient_scope for scope failures
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        token_verifier: TokenVerifier,
        ip_allowlist: list[IpNetwork] | None = None,
        resource_metadata_url: str | None = None,
        required_scopes: list[str] | None = None,
    ) -> None:
        self.app = app
        self.token_verifier = token_verifier
        self.ip_allowlist = ip_allowlist or []
        self.resource_metadata_url = resource_metadata_url
        self.required_scopes = required_scopes or []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Check IP allowlist first
        client_ip = _get_client_ip(scope)
        if (
            client_ip
            and self.ip_allowlist
            and ip_in_allowlist(client_ip, self.ip_allowlist)
        ):
            logger.debug("IP %s in allowlist, skipping auth", client_ip)
            await self.app(scope, receive, send)
            return

        # Extract Bearer token from Authorization header
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()

        if not auth_header or not auth_header.lower().startswith("bearer "):
            await self._send_401(scope, send, "Authentication required")
            return

        token = auth_header[7:]  # Strip "Bearer " prefix
        access_token = await self.token_verifier.verify_token(token)

        if access_token is None:
            await self._send_401(scope, send, "Invalid or expired token")
            return

        # Check expiry
        if access_token.expires_at and access_token.expires_at < int(time.time()):
            await self._send_401(scope, send, "Token expired")
            return

        # Check scopes
        for required in self.required_scopes:
            if required not in access_token.scopes:
                await self._send_403(scope, send, f"Required scope: {required}")
                return

        await self.app(scope, receive, send)

    async def _send_401(self, scope: Scope, send: Send, description: str) -> None:
        """401 Unauthorized with WWW-Authenticate per MCP spec / RFC 6750 §3."""
        if scope["type"] == "websocket":
            await _deny_websocket(send)
            return

        parts = ['error="invalid_token"', f'error_description="{description}"']
        if self.resource_metadata_url:
            parts.append(f'resource_metadata="{self.resource_metadata_url}"')
        if self.required_scopes:
            parts.append(f'scope="{" ".join(self.required_scopes)}"')

        www_auth = f"Bearer {', '.join(parts)}"
        body = json.dumps(
            {"error": "invalid_token", "error_description": description}
        ).encode()

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", www_auth.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _send_403(self, scope: Scope, send: Send, description: str) -> None:
        """403 Forbidden with insufficient_scope per MCP spec / RFC 6750 §3.1."""
        if scope["type"] == "websocket":
            await _deny_websocket(send)
            return

        parts = ['error="insufficient_scope"', f'error_description="{description}"']
        if self.resource_metadata_url:
            parts.append(f'resource_metadata="{self.resource_metadata_url}"')
        if self.required_scopes:
            parts.append(f'scope="{" ".join(self.required_scopes)}"')

        www_auth = f"Bearer {', '.join(parts)}"
        body = json.dumps(
            {"error": "insufficient_scope", "error_description": description}
        ).encode()

        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", www_auth.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
