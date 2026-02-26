import click
from pathlib import Path
import logging
import sys
from .server import serve
from .utils import find_lammps_binary


@click.command()
@click.option("--lammps-binary", "-b", help="Path to LAMMPS executable")
@click.option(
    "--working-directory",
    "-w",
    type=Path,
    default=Path.cwd(),
    help="Working directory for simulations",
)
@click.option("--remote", is_flag=True, help="Enable remote mode (SSE)")
@click.option("--host", default="0.0.0.0", help="Host for remote mode")
@click.option("--port", default=8000, help="Port for remote mode")
@click.option("-v", "--verbose", count=True)
def main(
    lammps_binary: str,
    working_directory: Path,
    remote: bool,
    host: str,
    port: int,
    verbose: bool
) -> None:
    """MCP LAMMPS Server - LAMMPS functionality for MCP"""
    import asyncio

    if not lammps_binary:
        lammps_binary = find_lammps_binary() or "lmp"

    logging_level = logging.WARN
    if verbose == 1:
        logging_level = logging.INFO
    elif verbose >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(level=logging_level, stream=sys.stderr)
    asyncio.run(serve(lammps_binary, working_directory, remote=remote, host=host, port=port))


if __name__ == "__main__":
    main()
