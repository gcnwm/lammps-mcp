import logging
from pathlib import Path
from typing import Optional, Sequence
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)
from enum import Enum
from pydantic import BaseModel, Field
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class LammpsSubmitScript(BaseModel):
    script_content: str = Field(
        ..., description="The content of the LAMMPS input script to execute."
    )
    script_name: str = Field(
        "in.lammps", description="The name to save the input script as."
    )
    log_file: str = Field(
        "log.lammps", description="The name of the log file to generate."
    )


class LammpsReadLog(BaseModel):
    log_file: str = Field(
        "log.lammps",
        description="Path to the log file to read (relative to working directory or latest archive).",
    )
    extract_performance: bool = Field(
        True, description="If true, extracts and returns performance/timing data."
    )


class LammpsReadOutput(BaseModel):
    filepath: str = Field(
        ...,
        description="Path to the output file to read (relative to working directory or latest archive).",
    )


class LammpsRestart(BaseModel):
    restart_file: str = Field(..., description="Path to the binary restart file.")
    action: str = Field(
        "data",
        description="Action to perform: 'data' (convert to data file), 'info' (get restart info), 'dump' (convert to dump).",
    )
    output_file: Optional[str] = Field(
        None, description="Path to the output file (required for 'data' and 'dump')."
    )


class LammpsTools(str, Enum):
    SUBMIT_SCRIPT = "submit_script"
    READ_LOG = "read_log"
    READ_OUTPUT = "read_output"
    RESTART = "restart"


def validate_path(path_str: str, working_dir: Path) -> Path:
    working_dir = working_dir.resolve()
    path_obj = Path(path_str)
    try:
        if path_obj.is_absolute():
            path = path_obj.resolve()
        else:
            path = (working_dir / path_obj).resolve()
        path.relative_to(working_dir)
        return path
    except (ValueError, RuntimeError):
        raise ValueError(f"Path '{path_str}' is outside the working directory")


def find_latest_archive(working_dir: Path) -> Optional[Path]:
    archive_base = working_dir / "archives"
    if not archive_base.exists():
        return None
    dirs = sorted([d for d in archive_base.iterdir() if d.is_dir()], reverse=True)
    return dirs[0] if dirs else None


def find_latest_run_directory(working_dir: Path) -> Optional[Path]:
    run_pattern = re.compile(r"^\d{8}_\d{6}(?:_\d{6})?_.+")
    run_dirs = sorted(
        [
            d
            for d in working_dir.iterdir()
            if d.is_dir() and d.name != "archives" and run_pattern.match(d.name)
        ],
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


async def run_optimized_lammps(
    binary: str, working_dir: Path, script_content: str, script_name: str, log_file: str
) -> str:
    from .utils import optimize_lammps_command

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_file_name = Path(script_name).name or "in.lammps"
    safe_script_name = re.sub(r"[^A-Za-z0-9._-]+", "_", script_file_name)
    run_dir = working_dir / f"{timestamp}_{safe_script_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    script_path = validate_path(script_file_name, run_dir)
    log_file_name = Path(log_file).name or "log.lammps"
    log_path = validate_path(log_file_name, run_dir)

    # Save script
    with open(script_path, "w") as f:
        f.write(script_content)

    # Get optimized base command
    base_cmd = optimize_lammps_command(binary)

    # Construct full command
    cmd = list(base_cmd)
    cmd.extend(["-in", script_file_name, "-log", log_file_name])

    try:
        import asyncio

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(run_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode()
        stderr_text = stderr.decode()

        with open(run_dir / "stdout.log", "w") as f:
            f.write(stdout_text)
        with open(run_dir / "stderr.log", "w") as f:
            f.write(stderr_text)

        if process.returncode != 0:
            raise RuntimeError(
                f"Simulation failed with code {process.returncode}.\n"
                f"Stderr:\n{stderr_text}\nStdout head:\n{stdout_text[:500]}"
            )

        run_dir_relative = run_dir.relative_to(working_dir)
        output = (
            f"Simulation completed successfully.\n"
            f"Command used: {' '.join(cmd)}\n"
            f"Run directory: {run_dir_relative}\n\n"
        )
        output += parse_thermo_from_log(log_path, extract_performance=True)
        return output
    except Exception as e:
        raise RuntimeError(f"Simulation error: {str(e)}")


def parse_thermo_from_log(log_path: Path, extract_performance: bool) -> str:
    if not log_path.exists():
        return f"Log file {log_path} not found."
    try:
        with open(log_path, "r") as f:
            content = f.read()
        output_parts = []
        lines = content.split("\n")
        data_blocks = []
        current_block = []
        capture = False
        headers = []
        for line in lines:
            ls = line.strip()
            if not ls:
                continue
            if ls.startswith("Step"):
                if capture and current_block:
                    data_blocks.append((headers, current_block))
                capture = True
                headers = ls.split()
                current_block = []
                continue
            if ls.startswith("Loop time"):
                if capture and current_block:
                    data_blocks.append((headers, current_block))
                capture = False
                continue
            if capture:
                parts = ls.split()
                if len(parts) == len(headers):
                    try:
                        current_block.append([p.replace("--", "-") for p in parts])
                    except ValueError:
                        pass
        if data_blocks:
            output_parts.append("Thermodynamic Data Summary:")
            for i, (head, block) in enumerate(data_blocks):
                output_parts.append(f"Run {i + 1}: {', '.join(head)}")
                if block:
                    output_parts.append(f"  Final State: {' '.join(block[-1])}")
        if extract_performance:
            matches = re.findall(
                r"(Loop time of.*?)(?=\n\s*\n|Step|$)", content, re.DOTALL
            )
            if matches:
                output_parts.append("\nPerformance Summary:")
                for i, m in enumerate(matches):
                    output_parts.append(f"Run {i + 1}:\n{m.strip()}")
        return "\n".join(output_parts) or "No relevant data found in log."
    except Exception as e:
        return f"Error parsing log: {str(e)}"


def create_server(lammps_binary: str, working_directory: Path) -> Server:
    """Create and configure the MCP LAMMPS server with all tool handlers.

    This is the single source of truth for tool registration.  Both stdio
    and HTTP transports call this function so the handler logic is never
    duplicated.
    """
    server = Server("mcp-lammps")
    working_directory = working_directory.expanduser().resolve()
    working_directory.mkdir(parents=True, exist_ok=True)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=LammpsTools.SUBMIT_SCRIPT,
                description="Submit a LAMMPS input script for automated execution.",
                inputSchema=LammpsSubmitScript.model_json_schema(),
            ),
            Tool(
                name=LammpsTools.READ_LOG,
                description="Extract thermodynamic and performance data from a log file.",
                inputSchema=LammpsReadLog.model_json_schema(),
            ),
            Tool(
                name=LammpsTools.READ_OUTPUT,
                description="Read the content of an output file (data, dump, custom).",
                inputSchema=LammpsReadOutput.model_json_schema(),
            ),
            Tool(
                name=LammpsTools.RESTART,
                description="Manage binary restart files (convert to data/dump or get info).",
                inputSchema=LammpsRestart.model_json_schema(),
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[TextContent] | CallToolResult:
        match name:
            case LammpsTools.SUBMIT_SCRIPT:
                args = LammpsSubmitScript(**arguments)
                try:
                    res = await run_optimized_lammps(
                        lammps_binary,
                        working_directory,
                        args.script_content,
                        args.script_name,
                        args.log_file,
                    )
                except RuntimeError as e:
                    return CallToolResult(
                        content=[TextContent(type="text", text=str(e))],
                        isError=True,
                    )
                return [TextContent(type="text", text=res)]

            case LammpsTools.READ_LOG:
                args = LammpsReadLog(**arguments)

                try:
                    path = validate_path(args.log_file, working_directory)
                except ValueError as e:
                    return CallToolResult(
                        content=[TextContent(type="text", text=str(e))],
                        isError=True,
                    )

                if not path.exists():
                    latest_run = find_latest_run_directory(working_directory)
                    if latest_run:
                        latest_run_path = latest_run / args.log_file
                        if latest_run_path.exists():
                            path = latest_run_path
                        else:
                            latest_run_name_path = latest_run / Path(args.log_file).name
                            if latest_run_name_path.exists():
                                path = latest_run_name_path
                if not path.exists():
                    latest = find_latest_archive(working_directory)
                    if latest and (latest / args.log_file).exists():
                        path = latest / args.log_file
                res = parse_thermo_from_log(path, args.extract_performance)
                return [TextContent(type="text", text=res)]

            case LammpsTools.READ_OUTPUT:
                args = LammpsReadOutput(**arguments)

                try:
                    path = validate_path(args.filepath, working_directory)
                except ValueError as e:
                    return CallToolResult(
                        content=[TextContent(type="text", text=str(e))],
                        isError=True,
                    )

                if not path.exists():
                    latest_run = find_latest_run_directory(working_directory)
                    if latest_run:
                        latest_run_path = latest_run / args.filepath
                        if latest_run_path.exists():
                            path = latest_run_path
                        else:
                            latest_run_name_path = latest_run / Path(args.filepath).name
                            if latest_run_name_path.exists():
                                path = latest_run_name_path
                if not path.exists():
                    latest = find_latest_archive(working_directory)
                    if latest and (latest / args.filepath).exists():
                        path = latest / args.filepath
                if not path.exists():
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"File {args.filepath} not found.",
                            )
                        ],
                        isError=True,
                    )
                with open(path, "r") as f:
                    return [
                        TextContent(
                            type="text",
                            text=f.read(10000)
                            + (
                                "\n...(truncated)"
                                if path.stat().st_size > 10000
                                else ""
                            ),
                        )
                    ]

            case LammpsTools.RESTART:
                args = LammpsRestart(**arguments)

                # Security: Prevent path traversal by validating input/output files
                try:
                    path = validate_path(args.restart_file, working_directory)
                except ValueError as e:
                    return CallToolResult(
                        content=[TextContent(type="text", text=str(e))],
                        isError=True,
                    )

                cmd = [lammps_binary]
                if args.action == "data":
                    if not args.output_file:
                        return CallToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text="output_file is required for 'data' action.",
                                )
                            ],
                            isError=True,
                        )

                    try:
                        out_path = validate_path(args.output_file, working_directory)
                    except ValueError as e:
                        return CallToolResult(
                            content=[TextContent(type="text", text=str(e))],
                            isError=True,
                        )

                    cmd.extend(["-restart2data", str(path), str(out_path)])
                elif args.action == "info":
                    cmd.extend(["-restart2info", str(path)])
                elif args.action == "dump":
                    if not args.output_file:
                        return CallToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text="output_file is required for 'dump' action.",
                                )
                            ],
                            isError=True,
                        )

                    try:
                        out_path = validate_path(args.output_file, working_directory)
                    except ValueError as e:
                        return CallToolResult(
                            content=[TextContent(type="text", text=str(e))],
                            isError=True,
                        )

                    cmd.extend(
                        ["-restart2dump", str(path), "all", "atom", str(out_path)]
                    )
                else:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Unknown action: {args.action}. Use 'data', 'info', or 'dump'.",
                            )
                        ],
                        isError=True,
                    )

                try:
                    import asyncio

                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=str(working_directory),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await process.communicate()
                    if process.returncode != 0:
                        return CallToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text=f"Action '{args.action}' failed (code {process.returncode}).\n"
                                    f"Stderr: {stderr.decode()}",
                                )
                            ],
                            isError=True,
                        )
                    return [
                        TextContent(
                            type="text",
                            text=f"Action '{args.action}' completed.\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}",
                        )
                    ]
                except Exception as e:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Restart error: {str(e)}",
                            )
                        ],
                        isError=True,
                    )

            case _:
                raise ValueError(f"Unknown tool: {name}")

    return server


# ---------------------------------------------------------------------------
# Transport entry-points
# ---------------------------------------------------------------------------


async def serve_stdio(lammps_binary: str, working_directory: Path) -> None:
    """Run the MCP server over stdio (default mode for local clients)."""
    server = create_server(lammps_binary, working_directory)
    options = server.create_initialization_options()
    async with stdio_server() as (r, w):
        await server.run(r, w, options, raise_exceptions=True)


async def serve_http(
    lammps_binary: str,
    working_directory: Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    skip_auth: bool = False,
    allowed_ips: Sequence[str] | None = None,
) -> None:
    """Run the MCP server over StreamableHTTP with optional auth.

    Auth modes (mutually-exclusive precedence):
      1. ``--skip-auth``  → no authentication at all
      2. ``--allowed-ips`` without Bearer → IPs in the allowlist pass through;
         others must present a valid Bearer token
      3. Default → Bearer token required for every request
    """
    import sys
    import uvicorn
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.routing import Route

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.auth.middleware.bearer_auth import (
        BearerAuthBackend,
        RequireAuthMiddleware,
    )
    from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
    from mcp.server.auth.routes import (
        build_resource_metadata_url,
        create_protected_resource_routes,
    )
    from starlette.middleware.authentication import AuthenticationMiddleware
    from pydantic import AnyHttpUrl

    from .auth import (
        StaticTokenVerifier,
        CombinedAuthMiddleware,
        parse_ip_allowlist,
    )

    server = create_server(lammps_binary, working_directory)
    session_manager = StreamableHTTPSessionManager(app=server)

    # ---- Auth setup --------------------------------------------------------
    base_url = f"http://{host}:{port}"
    resource_server_url = AnyHttpUrl(base_url)
    resource_metadata_url = build_resource_metadata_url(resource_server_url)

    ip_allowlist = parse_ip_allowlist(allowed_ips) if allowed_ips else None
    token_verifier: StaticTokenVerifier | None = None

    routes: list[Route] = []
    auth_middleware_stack: list[tuple] = []

    if skip_auth:
        # ── Mode 1: no auth at all ──────────────────────────────────────
        logger.info("Auth DISABLED (--skip-auth)")

        # Wrap in a class so Starlette treats it as a raw ASGI app
        # (bound methods are seen as request-response handlers → GET only).
        class _NoAuth:
            async def __call__(self, scope, receive, send):
                await session_manager.handle_request(scope, receive, send)

        routes.append(
            Route("/mcp", endpoint=_NoAuth()),
        )
    else:
        # Generate a static Bearer token
        token_verifier = StaticTokenVerifier()

        # SDK-level middleware: parses Authorization header → scope["user"]
        auth_middleware_stack = [
            (AuthenticationMiddleware, {"backend": BearerAuthBackend(token_verifier)}),
            (AuthContextMiddleware, {}),
        ]

        if ip_allowlist:
            # ── Mode 2: IP allowlist + Bearer fallback ──────────────────
            logger.info(
                "Auth: IP allowlist active (%d networks); "
                "non-listed IPs require Bearer token",
                len(ip_allowlist),
            )
            # Wrap the handler with our CombinedAuthMiddleware so that
            # whitelisted IPs bypass the SDK's RequireAuthMiddleware.
            combined = CombinedAuthMiddleware(
                app=session_manager.handle_request,
                token_verifier=token_verifier,
                ip_allowlist=ip_allowlist,
                resource_metadata_url=str(resource_metadata_url),
                required_scopes=["mcp:full"],
            )
            routes.append(Route("/mcp", endpoint=combined))
        else:
            # ── Mode 3: Bearer token required for everyone ──────────────
            logger.info("Auth: Bearer token required")
            routes.append(
                Route(
                    "/mcp",
                    endpoint=RequireAuthMiddleware(
                        session_manager.handle_request,
                        required_scopes=["mcp:full"],
                        resource_metadata_url=resource_metadata_url,
                    ),
                ),
            )

        # RFC 9728 Protected Resource Metadata endpoint
        routes.extend(
            create_protected_resource_routes(
                resource_url=resource_server_url,
                authorization_servers=[resource_server_url],  # self-issued
                scopes_supported=["mcp:full"],
            )
        )

    app = Starlette(
        debug=False,
        routes=routes,
        middleware=[],
        lifespan=lambda _app: session_manager.run(),
    )

    # Apply auth middleware as direct ASGI wrapping (avoids Middleware() type issues)
    for mw_cls, mw_kwargs in reversed(auth_middleware_stack):
        app = mw_cls(app, **mw_kwargs)
    app = CORSMiddleware(
        app=app,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Print connection info to stderr so operators can see it
    if token_verifier and not skip_auth:
        print(
            f"MCP StreamableHTTP server listening on {base_url}/mcp\n"
            f"Bearer token: {token_verifier.token}\n"
            f"Protected Resource Metadata: {resource_metadata_url}",
            file=sys.stderr,
        )
    else:
        print(
            f"MCP StreamableHTTP server listening on {base_url}/mcp (NO AUTH)",
            file=sys.stderr,
        )

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()
