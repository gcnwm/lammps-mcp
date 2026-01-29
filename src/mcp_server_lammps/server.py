import logging
from pathlib import Path
from typing import Sequence, Optional, Any, Dict, List, Union
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)
from enum import Enum
from pydantic import BaseModel, Field
import subprocess
import os
import re
import shlex

logger = logging.getLogger(__name__)

class LammpsRun(BaseModel):
    input_file: str = Field(..., description="Path to the input script file relative to working directory")
    log_file: Optional[str] = Field("log.lammps", description="Path to the output log file (uses -log)")
    screen_file: Optional[str] = Field(None, description="Path to the screen output file (uses -screen). Set to 'none' to suppress.")
    variables: Optional[Dict[str, str]] = Field(None, description="Dictionary of variables to define (uses -var name value)")
    suffix: Optional[str] = Field(None, description="Suffix style to use (uses -sf, e.g., 'omp', 'gpu')")
    package: Optional[List[str]] = Field(None, description="Package command arguments (uses -pk, e.g., ['omp', '4'])")
    options: Optional[List[str]] = Field(None, description="Additional raw command line options")

class LammpsReadLog(BaseModel):
    log_file: str = Field(..., description="Path to the log file to read")
    get_all_steps: bool = Field(False, description="If true, returns all steps. If false, returns only the final thermo data.")
    extract_performance: bool = Field(False, description="If true, extracts and returns performance/timing data.")

# class LammpsValidate(BaseModel):
#     input_file: str = Field(..., description="Path to the input script file to validate")

class LammpsRestart2Data(BaseModel):
    restart_file: str = Field(..., description="Path to the binary restart file")
    data_file: str = Field(..., description="Path to the output data file")

class LammpsTools(str, Enum):
    RUN = "run"
    READ_LOG = "read_log"
    # VALIDATE = "validate"
    RESTART2DATA = "restart2data"

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

def run_lammps(binary_cmd: List[str], working_dir: Path, args: LammpsRun) -> str:
    input_path = validate_path(args.input_file, working_dir)
    
    # Construct command
    cmd = list(binary_cmd)
    cmd.extend(["-in", str(input_path)])
    
    if args.log_file:
        log_path = validate_path(args.log_file, working_dir)
        cmd.extend(["-log", str(log_path)])
    
    if args.screen_file:
        if args.screen_file.lower() == "none":
            cmd.extend(["-screen", "none"])
        else:
            screen_path = validate_path(args.screen_file, working_dir)
            cmd.extend(["-screen", str(screen_path)])

    if args.variables:
        for key, value in args.variables.items():
            cmd.extend(["-var", str(key), str(value)])
            
    if args.suffix:
        cmd.extend(["-suffix", args.suffix])
        
    if args.package:
        cmd.append("-package")
        cmd.extend(args.package)

    if args.options:
        cmd.extend(args.options)
        
    logging.info(f"Executing LAMMPS: {' '.join(cmd)}")
    
    try:
        # Run in the working directory so relative paths in input script work
        result = subprocess.run(
            cmd, 
            cwd=str(working_dir),
            capture_output=True, 
            text=True, 
            check=True
        )
        output = "Simulation completed successfully."
        if args.log_file and args.log_file != "none":
            output += f"\nLog written to {args.log_file}"
        output += f"\nOutput head:\n{result.stdout[:1000]}..."
        return output
    except subprocess.CalledProcessError as e:
        return f"Simulation failed with exit code {e.returncode}.\nStderr:\n{e.stderr}\nStdout:\n{e.stdout}"
    except FileNotFoundError:
        return f"LAMMPS binary command '{binary_cmd}' failed. Executable not found."

def convert_restart2data(binary_cmd: List[str], working_dir: Path, restart_file: str, data_file: str) -> str:
    restart_path = validate_path(restart_file, working_dir)
    data_path = validate_path(data_file, working_dir)
    
    # lmp -restart2data restartfile datafile
    cmd = list(binary_cmd)
    cmd.extend(["-restart2data", str(restart_path), str(data_path)])
    
    logging.info(f"Executing Restart2Data: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            check=True
        )
        return f"Conversion successful.\nData file written to {data_file}\nOutput:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Conversion failed with exit code {e.returncode}.\nStderr:\n{e.stderr}\nStdout:\n{e.stdout}"

# def validate_script(binary_cmd: List[str], working_dir: Path, input_file: str) -> str:
#     input_path = validate_path(input_file, working_dir)
    
#     cmd = list(binary_cmd)
#     cmd.extend(["-in", str(input_path), "-check", "on"])
    
#     try:
#         result = subprocess.run(
#             cmd,
#             cwd=str(working_dir),
#             capture_output=True,
#             text=True
#         )
#         if result.returncode == 0:
#             return "Syntax check passed (based on exit code 0)."
#         else:
#              return f"Syntax check output:\n{result.stdout}\n{result.stderr}"
#     except FileNotFoundError:
#          return "Executable not found."

def parse_log(working_dir: Path, log_file: str, all_steps: bool, extract_performance: bool) -> str:
    log_path = validate_path(log_file, working_dir)
    if not log_path.exists():
        return f"Log file {log_file} not found."
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
        output_parts = []
        
        # 1. Thermo Data Parsing
        lines = content.split('\n')
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
                        # check if numeric (basic check)
                        [float(x) for x in parts]
                        current_block.append(parts)
                    except ValueError:
                        pass
        
        if data_blocks:
            output_parts.append("Thermodynamic Data:")
            for i, (head, block) in enumerate(data_blocks):
                output_parts.append(f"Run {i+1}:")
                output_parts.append(f"  Fields: {', '.join(head)}")
                if all_steps:
                    output_parts.append("  Data:")
                    for row in block:
                        output_parts.append("    " + " ".join(row))
                else:
                    if block:
                        output_parts.append(f"  Final Step: {' '.join(block[-1])}")
                    else:
                        output_parts.append("  No steps recorded.")
        else:
            output_parts.append("No thermodynamic data found.")

        # 2. Performance Data Parsing
        if extract_performance:
            output_parts.append("\nPerformance Data:")
            # Look for "Loop time" and subsequent lines
            # Pattern: Loop time of ...
            #          Performance: ...
            #          ... CPU use ...
            #          MPI task timing breakdown: ...
            # This is unstructured text, so we'll grab lines from "Loop time" until next empty line or specific marker?
            # Actually, usually it's at the end of a run.
            
            perf_regex = re.compile(r"(Loop time of.*?)(?=\n\s*\n|Step|$)", re.DOTALL)
            matches = perf_regex.findall(content)
            
            if matches:
                for i, match in enumerate(matches):
                    output_parts.append(f"Run {i+1} Summary:\n{match.strip()}")
            else:
                 output_parts.append("No performance summary found.")

        return "\n".join(output_parts)
            
    except Exception as e:
        return f"Error reading log file: {str(e)}"

async def serve(lammps_binary: str, working_directory: Path) -> None:
    server = Server("mcp-lammps")
    
    # Resolve and ensure working dir exists
    working_directory = working_directory.expanduser().resolve()
    if not working_directory.exists():
        logger.warning(f"Working directory {working_directory} does not exist. Creating it.")
        working_directory.mkdir(parents=True, exist_ok=True)
    
    # Parse the binary string into a command list (handles spaces/quotes)
    # e.g. "mpiexec -np 4 lmp" -> ["mpiexec", "-np", "4", "lmp"]
    # On Windows, shlex.split might need posix=False for backslashes, but we are using forward slashes or user should escape.
    # Actually, for simple commands it's fine.
    binary_cmd = shlex.split(lammps_binary, posix=(os.name != 'nt'))
    
    logger.info(f"Starting LAMMPS MCP server at {working_directory}")
    logger.info(f"Using LAMMPS command: {binary_cmd}")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=LammpsTools.RUN,
                description="Run a LAMMPS simulation with various options (input file, variables, packages)",
                inputSchema=LammpsRun.model_json_schema(),
            ),
            Tool(
                name=LammpsTools.READ_LOG,
                description="Read thermodynamic and performance data from a LAMMPS log file",
                inputSchema=LammpsReadLog.model_json_schema(),
            ),
            # Tool(
            #     name=LammpsTools.VALIDATE,
            #     description="Validate a LAMMPS input script (syntax check only)",
            #     inputSchema=LammpsValidate.model_json_schema(),
            # ),
            Tool(
                name=LammpsTools.RESTART2DATA,
                description="Convert a binary restart file to a text data file",
                inputSchema=LammpsRestart2Data.model_json_schema(),
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        match name:
            case LammpsTools.RUN:
                args = LammpsRun(**arguments)
                result = run_lammps(binary_cmd, working_directory, args)
                return [TextContent(type="text", text=result)]
                
            case LammpsTools.READ_LOG:
                args = LammpsReadLog(**arguments)
                result = parse_log(
                    working_directory,
                    args.log_file,
                    args.get_all_steps,
                    args.extract_performance
                )
                return [TextContent(type="text", text=result)]
                
            # case LammpsTools.VALIDATE:
            #     result = validate_script(
            #         binary_cmd,
            #         working_directory,
            #         arguments["input_file"]
            #     )
            #     return [TextContent(type="text", text=result)]
            
            case LammpsTools.RESTART2DATA:
                result = convert_restart2data(
                    binary_cmd,
                    working_directory,
                    arguments["restart_file"],
                    arguments["data_file"]
                )
                return [TextContent(type="text", text=result)]
                
            case _:
                raise ValueError(f"Unknown tool: {name}")

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)
