import click
from pathlib import Path
import logging
import sys
from .server import serve_stdio, serve_http
from .utils import find_lammps_binary


def _setup_logging(verbose: int) -> None:
    level = logging.WARN
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, stream=sys.stderr)


@click.group(invoke_without_command=True)
@click.option("--lammps-binary", "-b", help="Path to LAMMPS executable")
@click.option(
    "--working-directory",
    "-w",
    type=Path,
    default=Path.cwd(),
    help="Working directory for simulations",
)
@click.option("-v", "--verbose", count=True)
@click.pass_context
def main(
    ctx: click.Context, lammps_binary: str, working_directory: Path, verbose: int
) -> None:
    """MCP LAMMPS Server - LAMMPS functionality for MCP

    By default runs in stdio mode (for local MCP clients like Claude Desktop).
    Use the 'serve' subcommand for remote HTTP deployment.
    """
    if not lammps_binary:
        lammps_binary = find_lammps_binary() or "lmp"

    ctx.ensure_object(dict)
    ctx.obj["lammps_binary"] = lammps_binary
    ctx.obj["working_directory"] = working_directory
    ctx.obj["verbose"] = verbose

    # If no subcommand was given, run in stdio mode (default)
    if ctx.invoked_subcommand is None:
        _setup_logging(verbose)
        import asyncio

        asyncio.run(serve_stdio(lammps_binary, working_directory))


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind address")
@click.option("--port", default=8000, show_default=True, help="Bind port")
@click.option(
    "--skip-auth",
    is_flag=True,
    default=False,
    help="Disable authentication entirely (not recommended for public networks)",
)
@click.option(
    "--allowed-ips",
    multiple=True,
    help="IP/CIDR allowlist entries that bypass auth (e.g. 192.168.0.0/16). "
    "Can be specified multiple times. "
    "Non-listed IPs still require a Bearer token unless --skip-auth is set.",
)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    skip_auth: bool,
    allowed_ips: tuple[str, ...],
) -> None:
    """Run the MCP server over StreamableHTTP for remote access.

    \b
    Examples:
      # Bearer token required (token printed to stderr at startup)
      uv run mcp-server-lammps serve --host 0.0.0.0 --port 8000

      # No authentication
      uv run mcp-server-lammps serve --skip-auth

      # LAN IPs bypass auth; others need Bearer token
      uv run mcp-server-lammps serve --allowed-ips 192.168.0.0/16 --allowed-ips 10.0.0.0/8
    """
    _setup_logging(ctx.obj["verbose"])
    import asyncio

    asyncio.run(
        serve_http(
            ctx.obj["lammps_binary"],
            ctx.obj["working_directory"],
            host=host,
            port=port,
            skip_auth=skip_auth,
            allowed_ips=list(allowed_ips) if allowed_ips else None,
        )
    )


if __name__ == "__main__":
    main()
