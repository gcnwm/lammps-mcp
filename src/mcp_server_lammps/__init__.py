import click
from pathlib import Path
import logging
import sys
from .server import serve


@click.command()
@click.option("--lammps-binary", "-b", default="lmp", help="Path to LAMMPS executable")
@click.option(
    "--working-directory",
    "-w",
    type=Path,
    default=Path.cwd(),
    help="Working directory for simulations",
)
@click.option("-v", "--verbose", count=True)
def main(lammps_binary: str, working_directory: Path, verbose: bool) -> None:
    """MCP LAMMPS Server - LAMMPS functionality for MCP"""
    import asyncio

    logging_level = logging.WARN
    if verbose == 1:
        logging_level = logging.INFO
    elif verbose >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(level=logging_level, stream=sys.stderr)
    asyncio.run(serve(lammps_binary, working_directory))


if __name__ == "__main__":
    main()
