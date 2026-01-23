import logging
from pathlib import Path
from typing import Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)
from enum import Enum
from pydantic import BaseModel, Field
import subprocess

logger = logging.getLogger(__name__)


class LammpsRun(BaseModel):
    input_file: str = Field(
        ..., description="Path to the input script file relative to working directory"
    )
    log_file: Optional[str] = Field(
        "log.lammps", description="Path to the output log file"
    )
    options: Optional[list[str]] = Field(
        None, description="Additional command line options for LAMMPS"
    )


class LammpsReadLog(BaseModel):
    log_file: str = Field(..., description="Path to the log file to read")
    get_all_steps: bool = Field(
        False,
        description="If true, returns all steps. If false, returns only the final thermo data.",
    )


class LammpsValidate(BaseModel):
    input_file: str = Field(
        ..., description="Path to the input script file to validate"
    )


class LammpsTools(str, Enum):
    RUN = "run"
    READ_LOG = "read_log"
    # VALIDATE = "lammps_validate"


def validate_path(path_str: str, working_dir: Path) -> Path:
    """Validate that the path is within the working directory."""
    # Resolve working directory once to avoid inconsistencies on Windows
    working_dir = working_dir.resolve()
    path_obj = Path(path_str)
    try:
        # If an absolute path is provided, use it directly; otherwise join with working_dir
        if path_obj.is_absolute():
            path = path_obj.resolve()
        else:
            path = (working_dir / path_obj).resolve()

        # Ensure the resulting path is inside the working directory
        path.relative_to(working_dir)
        return path
    except (ValueError, RuntimeError):
        raise ValueError(f"Path '{path_str}' is outside the working directory")


def run_lammps(
    binary: str,
    working_dir: Path,
    input_file: str,
    log_file: str = "log.lammps",
    options: Optional[list[str]] = None,
) -> str:
    logging.info(
        f"Running LAMMPS with input file: {input_file}, log file: {log_file}, options: {options}"
    )
    input_path = validate_path(input_file, working_dir)
    log_path = validate_path(log_file, working_dir)

    cmd = [binary, "-in", str(input_path), "-log", str(log_path)]
    if options:
        cmd.extend(options)

    try:
        # Run in the working directory so relative paths in input script work
        result = subprocess.run(
            cmd, cwd=str(working_dir), capture_output=True, text=True, check=True
        )
        return f"Simulation completed successfully.\nLog written to {log_file}\nOutput:\n{result.stdout[:1000]}..."
    except subprocess.CalledProcessError as e:
        return f"Simulation failed with exit code {e.returncode}.\nStderr:\n{e.stderr}\nStdout:\n{e.stdout}"
    except FileNotFoundError:
        return f"LAMMPS binary '{binary}' not found."


def validate_script(binary: str, working_dir: Path, input_file: str) -> str:
    input_path = validate_path(input_file, working_dir)
    # Use -check command line switch: syntax check only
    # Note: Not all LAMMPS versions support -check, but most modern ones do or handle it gracefully?
    # Actually, standard way to dry-run is harder. But `lmp -in file -no-run` (if valid) or just check.
    # LAMMPS doesn't have a standardized "dry-run" flag that works everywhere without running setup.
    # However, passing "-h" shows help.
    # A common trick is adding "quit" at the beginning, but we can't easily modify the file.
    # Let's try to run with a nonexistent flag to see if it parses args? No.
    # We will assume the user knows if their binary supports validation or we just try running with a very short run? No that's dangerous.
    # Let's stick to parsing or basic check.
    # Actually, many versions support "-check". Let's try that.

    cmd = [binary, "-in", str(input_path), "-check", "on"]
    try:
        result = subprocess.run(
            cmd, cwd=str(working_dir), capture_output=True, text=True
        )
        if result.returncode == 0:
            return "Syntax check passed (based on exit code 0)."
        else:
            # LAMMPS might return 0 even if just showing help, but usually errors return non-zero.
            return f"Syntax check output:\n{result.stdout}\n{result.stderr}"
    except FileNotFoundError:
        return f"LAMMPS binary '{binary}' not found."


def parse_log(working_dir: Path, log_file: str, all_steps: bool) -> str:
    log_path = validate_path(log_file, working_dir)
    if not log_path.exists():
        return f"Log file {log_file} not found."

    try:
        with open(log_path, "r") as f:
            content = f.read()

        # simple parsing: find sections starting with "Step"
        # This is a heuristic parser.
        lines = content.split("\n")
        data_blocks = []
        current_block = []
        capture = False
        headers = []

        for line in lines:
            if line.strip().startswith("Step"):
                capture = True
                headers = line.split()
                current_block = []
                continue
            if line.strip().startswith("Loop time"):
                capture = False
                if current_block:
                    data_blocks.append((headers, current_block))
                continue

            if capture:
                parts = line.split()
                if len(parts) == len(headers):
                    try:
                        # check if numeric
                        [float(x) for x in parts]
                        current_block.append(parts)
                    except ValueError:
                        pass

        if not data_blocks:
            return "No thermodynamic data found in log file."

        output = []
        for i, (head, block) in enumerate(data_blocks):
            output.append(f"Run {i + 1}:")
            output.append(f"  Fields: {', '.join(head)}")
            if all_steps:
                output.append("  Data:")
                for row in block:
                    output.append("    " + " ".join(row))
            else:
                if block:
                    output.append(f"  Final Step: {' '.join(block[-1])}")
                else:
                    output.append("  No steps recorded.")

        return "\n".join(output)

    except Exception as e:
        return f"Error reading log file: {str(e)}"


async def serve(lammps_binary: str, working_directory: Path) -> None:
    # Resolve and ensure working dir exists
    working_directory = working_directory.expanduser().resolve()

    if not working_directory.exists():
        logger.warning(
            f"Working directory {working_directory} does not exist. Creating it."
        )
        working_directory.mkdir(parents=True, exist_ok=True)

    server = Server("mcp-lammps")

    logger.info(
        f"Starting LAMMPS MCP server at {working_directory} using binary '{lammps_binary}'"
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=LammpsTools.RUN,
                description="Run a LAMMPS simulation using an input script",
                inputSchema=LammpsRun.model_json_schema(),
            ),
            Tool(
                name=LammpsTools.READ_LOG,
                description="Read thermodynamic data from a LAMMPS log file",
                inputSchema=LammpsReadLog.model_json_schema(),
            ),
            # Tool(
            #     name=LammpsTools.VALIDATE,
            #     description="Validate a LAMMPS input script (syntax check)",
            #     inputSchema=LammpsValidate.model_json_schema(),
            # ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        match name:
            case LammpsTools.RUN:
                result = run_lammps(
                    lammps_binary,
                    working_directory,
                    arguments["input_file"],
                    arguments.get("log_file", "log.lammps"),
                    arguments.get("options"),
                )
                return [TextContent(type="text", text=result)]

            case LammpsTools.READ_LOG:
                result = parse_log(
                    working_directory,
                    arguments["log_file"],
                    arguments.get("get_all_steps", False),
                )
                return [TextContent(type="text", text=result)]

            # case LammpsTools.VALIDATE:
            #     result = validate_script(
            #         lammps_binary,
            #         working_directory,
            #         arguments["input_file"]
            #     )
            #     return [TextContent(type="text", text=result)]

            case _:
                raise ValueError(f"Unknown tool: {name}")

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)
