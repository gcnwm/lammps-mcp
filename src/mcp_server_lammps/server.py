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
import shutil
from datetime import datetime
import secrets

logger = logging.getLogger(__name__)

class LammpsSubmitScript(BaseModel):
    script_content: str = Field(..., description="The content of the LAMMPS input script to execute.")
    script_name: str = Field("in.lammps", description="The name to save the input script as.")
    log_file: str = Field("log.lammps", description="The name of the log file to generate.")

class LammpsReadLog(BaseModel):
    log_file: str = Field("log.lammps", description="Path to the log file to read (relative to working directory or latest archive).")
    extract_performance: bool = Field(True, description="If true, extracts and returns performance/timing data.")

class LammpsReadOutput(BaseModel):
    filepath: str = Field(..., description="Path to the output file to read (relative to working directory or latest archive).")

class LammpsRestart(BaseModel):
    restart_file: str = Field(..., description="Path to the binary restart file.")
    action: str = Field("data", description="Action to perform: 'data' (convert to data file), 'info' (get restart info), 'dump' (convert to dump).")
    output_file: Optional[str] = Field(None, description="Path to the output file (required for 'data' and 'dump').")

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
        # Allow reading from archives if it exists
        if "archives" in path_str:
             return (working_dir / path_str).resolve()
        raise ValueError(f"Path '{path_str}' is outside the working directory")

def find_latest_archive(working_dir: Path) -> Optional[Path]:
    archive_base = working_dir / "archives"
    if not archive_base.exists():
        return None
    dirs = sorted([d for d in archive_base.iterdir() if d.is_dir()], reverse=True)
    return dirs[0] if dirs else None

async def run_optimized_lammps(binary: str, working_dir: Path, script_content: str, script_name: str, log_file: str) -> str:
    from .utils import optimize_lammps_command

    # Save script
    script_path = working_dir / script_name
    with open(script_path, "w") as f:
        f.write(script_content)

    # Get optimized base command
    base_cmd = optimize_lammps_command(binary)

    # Construct full command
    cmd = list(base_cmd)
    cmd.extend(["-in", script_name, "-log", log_file])
    
    try:
        import asyncio
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(working_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode()
        stderr_text = stderr.decode()

        # Archiving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = working_dir / "archives" / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Copy input and log
        shutil.copy2(script_path, archive_dir)
        log_path = working_dir / log_file
        if log_path.exists():
            shutil.copy2(log_path, archive_dir)

        with open(archive_dir / "stdout.log", "w") as f: f.write(stdout_text)
        with open(archive_dir / "stderr.log", "w") as f: f.write(stderr_text)

        if process.returncode != 0:
            return f"Simulation failed with code {process.returncode}.\nStderr:\n{stderr_text}\nStdout head:\n{stdout_text[:500]}"

        output = f"Simulation completed successfully.\nCommand used: {' '.join(cmd)}\nArchived to: archives/{timestamp}\n\n"
        output += parse_thermo_from_log(log_path, extract_performance=True)
        return output
    except Exception as e:
        return f"Simulation error: {str(e)}"

def parse_thermo_from_log(log_path: Path, extract_performance: bool) -> str:
    if not log_path.exists(): return f"Log file {log_path} not found."
    try:
        with open(log_path, 'r') as f: content = f.read()
        output_parts = []
        lines = content.split('\n')
        data_blocks = []
        current_block = []
        capture = False
        headers = []
        for line in lines:
            ls = line.strip()
            if not ls: continue
            if ls.startswith("Step"):
                if capture and current_block: data_blocks.append((headers, current_block))
                capture = True
                headers = ls.split()
                current_block = []
                continue
            if ls.startswith("Loop time"):
                if capture and current_block: data_blocks.append((headers, current_block))
                capture = False
                continue
            if capture:
                parts = ls.split()
                if len(parts) == len(headers):
                    try:
                        current_block.append([p.replace("--", "-") for p in parts])
                    except ValueError: pass
        if data_blocks:
            output_parts.append("Thermodynamic Data Summary:")
            for i, (head, block) in enumerate(data_blocks):
                output_parts.append(f"Run {i+1}: {', '.join(head)}")
                if block: output_parts.append(f"  Final State: {' '.join(block[-1])}")
        if extract_performance:
            matches = re.findall(r"(Loop time of.*?)(?=\n\s*\n|Step|$)", content, re.DOTALL)
            if matches:
                output_parts.append("\nPerformance Summary:")
                for i, m in enumerate(matches): output_parts.append(f"Run {i+1}:\n{m.strip()}")
        return "\n".join(output_parts) or "No relevant data found in log."
    except Exception as e: return f"Error parsing log: {str(e)}"

async def serve(lammps_binary: str, working_directory: Path, remote: bool = False, host: str = "0.0.0.0", port: int = 8000) -> None:
    server = Server("mcp-lammps")
    working_directory = working_directory.expanduser().resolve()
    working_directory.mkdir(parents=True, exist_ok=True)
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(name=LammpsTools.SUBMIT_SCRIPT, description="Submit a LAMMPS input script for automated execution.", inputSchema=LammpsSubmitScript.model_json_schema()),
            Tool(name=LammpsTools.READ_LOG, description="Extract thermodynamic and performance data from a log file.", inputSchema=LammpsReadLog.model_json_schema()),
            Tool(name=LammpsTools.READ_OUTPUT, description="Read the content of an output file (data, dump, custom).", inputSchema=LammpsReadOutput.model_json_schema()),
            Tool(name=LammpsTools.RESTART, description="Manage binary restart files (convert to data/dump or get info).", inputSchema=LammpsRestart.model_json_schema()),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        match name:
            case LammpsTools.SUBMIT_SCRIPT:
                args = LammpsSubmitScript(**arguments)
                res = await run_optimized_lammps(lammps_binary, working_directory, args.script_content, args.script_name, args.log_file)
                return [TextContent(type="text", text=res)]

            case LammpsTools.READ_LOG:
                args = LammpsReadLog(**arguments)
                path = validate_path(args.log_file, working_directory)
                if not path.exists():
                    latest = find_latest_archive(working_directory)
                    if latest and (latest / args.log_file).exists():
                        path = latest / args.log_file
                res = parse_thermo_from_log(path, args.extract_performance)
                return [TextContent(type="text", text=res)]

            case LammpsTools.READ_OUTPUT:
                args = LammpsReadOutput(**arguments)
                path = validate_path(args.filepath, working_directory)
                if not path.exists():
                     latest = find_latest_archive(working_directory)
                     if latest and (latest / args.filepath).exists():
                         path = latest / args.filepath
                if not path.exists(): return [TextContent(type="text", text=f"File {args.filepath} not found.")]
                with open(path, "r") as f:
                    return [TextContent(type="text", text=f.read(10000) + ("\n...(truncated)" if path.stat().st_size > 10000 else ""))]

            case LammpsTools.RESTART:
                args = LammpsRestart(**arguments)
                path = validate_path(args.restart_file, working_directory)
                cmd = [lammps_binary]
                if args.action == "data":
                    if not args.output_file: return [TextContent(type="text", text="output_file is required for 'data' action.")]
                    cmd.extend(["-restart2data", str(path), args.output_file])
                elif args.action == "info":
                    cmd.extend(["-restart2info", str(path)])
                elif args.action == "dump":
                    if not args.output_file: return [TextContent(type="text", text="output_file is required for 'dump' action.")]
                    cmd.extend(["-restart2dump", str(path), "all", "atom", args.output_file])

                try:
                    import asyncio
                    process = await asyncio.create_subprocess_exec(*cmd, cwd=str(working_directory), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    stdout, stderr = await process.communicate()
                    return [TextContent(type="text", text=f"Action '{args.action}' completed.\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}")]
                except Exception as e:
                    return [TextContent(type="text", text=f"Restart error: {str(e)}")]

            case _: raise ValueError(f"Unknown tool: {name}")

    options = server.create_initialization_options()
    if remote:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        import uvicorn
        token = secrets.token_urlsafe(16)
        sse = SseServerTransport("/messages")
        async def handle_sse(request):
            if request.query_params.get("token") != token: return JSONResponse({"error": "Unauthorized"}, status_code=401)
            async with sse.connect_scope(request.scope, request.receive, request._send):
                await server.run(sse.read_stream, sse.write_stream, options, raise_exceptions=True)
        async def handle_messages(request): await sse.handle_post_message(request.scope, request.receive, request._send)
        app = Starlette(debug=True, routes=[Route("/sse", endpoint=handle_sse), Route("/messages", endpoint=handle_messages, methods=["POST"])])
        print(f"SSE URL: http://{host}:{port}/sse?token={token}")
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        await uvicorn.Server(config).serve()
    else:
        async with stdio_server() as (r, w): await server.run(r, w, options, raise_exceptions=True)
