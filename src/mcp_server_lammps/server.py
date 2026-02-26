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

class LammpsRun(BaseModel):
    input_file: Optional[str] = Field(None, description="Path to the input script file relative to working directory (uses -in)")
    log_file: Optional[str] = Field("log.lammps", description="Path to the output log file (uses -log)")
    screen_file: Optional[str] = Field(None, description="Path to the screen output file (uses -screen). Set to 'none' to suppress.")
    variables: Optional[Dict[str, Union[str, List[str]]]] = Field(None, description="Dictionary of variables to define (uses -var name value1 value2 ...)")
    suffix: Optional[List[str]] = Field(None, description="Suffix style to use (uses -sf, e.g., ['omp'])")
    package: Optional[List[List[str]]] = Field(None, description="Package command arguments (uses -pk, e.g., [['omp', '4']])")

    echo: Optional[str] = Field(None, description="Echo style: none, screen, log, both (uses -echo)")
    kokkos: Optional[List[str]] = Field(None, description="Kokkos options (uses -kokkos on/off ...)")
    mdi: Optional[str] = Field(None, description="MDI flags (uses -mdi)")
    mpicolor: Optional[int] = Field(None, description="MPI color (uses -mpicolor)")
    cite: Optional[str] = Field(None, description="Citation style or filename (uses -cite)")
    nocite: bool = Field(False, description="Disable citation reminder (uses -nocite)")
    nonbuf: bool = Field(False, description="Turn off buffering for screen and logfile (uses -nonbuf)")
    partition: Optional[List[str]] = Field(None, description="Multi-partition mode settings (uses -partition)")
    plog: Optional[str] = Field(None, description="Base name for partition log files (uses -plog)")
    pscreen: Optional[str] = Field(None, description="Base name for partition screen files (uses -pscreen)")
    reorder: Optional[List[str]] = Field(None, description="Processor reordering (uses -reorder)")
    skiprun: bool = Field(False, description="Skip run and minimize commands for testing (uses -skiprun)")

    restart2data: Optional[List[str]] = Field(None, description="Convert restart to data (uses -r2data restartfile datafile ...)")
    restart2dump: Optional[List[str]] = Field(None, description="Convert restart to dump (uses -r2dump restartfile group-ID style file ...)")
    restart2info: Optional[List[str]] = Field(None, description="Output restart info (uses -r2info restartfile ...)")

    options: Optional[List[str]] = Field(None, description="Additional raw command line options")

class LammpsReadLog(BaseModel):
    log_file: str = Field(..., description="Path to the log file to read")
    get_all_steps: bool = Field(False, description="If true, returns all steps. If false, returns only the final thermo data.")
    extract_performance: bool = Field(False, description="If true, extracts and returns performance/timing data.")

class LammpsRestart2Data(BaseModel):
    restart_file: str = Field(..., description="Path to the binary restart file")
    data_file: str = Field(..., description="Path to the output data file")

class LammpsUploadFile(BaseModel):
    filepath: str = Field(..., description="Path to save the file relative to working directory")
    content: str = Field(..., description="Content of the file")

class LammpsReadFile(BaseModel):
    filepath: str = Field(..., description="Path to the file to read relative to working directory")

class LammpsListFiles(BaseModel):
    directory: str = Field(".", description="Directory to list relative to working directory")

class LammpsGetInfo(BaseModel):
    pass

class LammpsTools(str, Enum):
    RUN = "run"
    READ_LOG = "read_log"
    GET_THERMO = "get_thermo"
    RESTART2DATA = "restart2data"
    UPLOAD_FILE = "upload_file"
    READ_FILE = "read_file"
    LIST_FILES = "list_files"
    GET_INFO = "get_info"

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

async def run_lammps(binary_cmd: List[str], working_dir: Path, args: LammpsRun) -> str:
    cmd = list(binary_cmd)
    if args.input_file:
        input_path = validate_path(args.input_file, working_dir)
        cmd.extend(["-in", str(input_path)])
    if args.log_file:
        if args.log_file.lower() == "none":
            cmd.extend(["-log", "none"])
        else:
            log_path = validate_path(args.log_file, working_dir)
            cmd.extend(["-log", str(log_path)])
    if args.screen_file:
        if args.screen_file.lower() == "none":
            cmd.extend(["-screen", "none"])
        else:
            screen_path = validate_path(args.screen_file, working_dir)
            cmd.extend(["-screen", str(screen_path)])
    if args.variables:
        for key, values in args.variables.items():
            cmd.append("-var")
            cmd.append(str(key))
            if isinstance(values, list):
                cmd.extend([str(v) for v in values])
            else:
                cmd.append(str(values))
    if args.suffix:
        cmd.append("-suffix")
        cmd.extend(args.suffix)
    if args.package:
        for pkg_args in args.package:
            cmd.append("-package")
            cmd.extend(pkg_args)
    if args.echo: cmd.extend(["-echo", args.echo])
    if args.kokkos:
        cmd.append("-kokkos")
        cmd.extend(args.kokkos)
    if args.mdi: cmd.extend(["-mdi", args.mdi])
    if args.mpicolor is not None: cmd.extend(["-mpicolor", str(args.mpicolor)])
    if args.cite: cmd.extend(["-cite", args.cite])
    if args.nocite: cmd.append("-nocite")
    if args.nonbuf: cmd.append("-nonbuf")
    if args.partition:
        cmd.append("-partition")
        cmd.extend(args.partition)
    if args.plog: cmd.extend(["-plog", args.plog])
    if args.pscreen: cmd.extend(["-pscreen", args.pscreen])
    if args.reorder:
        cmd.append("-reorder")
        cmd.extend(args.reorder)
    if args.skiprun: cmd.append("-skiprun")
    if args.restart2data:
        cmd.append("-restart2data")
        cmd.extend(args.restart2data)
    if args.restart2dump:
        cmd.append("-restart2dump")
        cmd.extend(args.restart2dump)
    if args.restart2info:
        cmd.append("-restart2info")
        cmd.extend(args.restart2info)
    if args.options: cmd.extend(args.options)
    
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

        if process.returncode != 0:
            return f"Simulation failed with code {process.returncode}.\nStderr:\n{stderr_text}\nStdout:\n{stdout_text}"

        output = "Simulation completed successfully."
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = working_dir / "archives" / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)
        if args.input_file:
            try: shutil.copy2(working_dir / args.input_file, archive_dir)
            except Exception: pass
        if args.log_file and args.log_file.lower() != "none":
            try:
                p = working_dir / args.log_file
                if p.exists(): shutil.copy2(p, archive_dir)
            except Exception: pass
        with open(archive_dir / "stdout.log", "w") as f: f.write(stdout_text)
        with open(archive_dir / "stderr.log", "w") as f: f.write(stderr_text)
        output += f"\nArchived to {archive_dir.relative_to(working_dir)}"
        return output
    except Exception as e:
        return f"Simulation error: {str(e)}"

async def convert_restart2data(binary_cmd: List[str], working_dir: Path, restart_file: str, data_file: str) -> str:
    restart_path = validate_path(restart_file, working_dir)
    data_path = validate_path(data_file, working_dir)
    cmd = list(binary_cmd) + ["-restart2data", str(restart_path), str(data_path)]
    try:
        import asyncio
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(working_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            return f"Conversion failed.\n{stderr.decode()}"
        return f"Conversion successful.\n{stdout.decode()}"
    except Exception as e:
        return f"Conversion error: {str(e)}"

def upload_file(working_dir: Path, filepath: str, content: str) -> str:
    target_path = validate_path(filepath, working_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w") as f: f.write(content)
    return f"File uploaded to {filepath}"

def read_file(working_dir: Path, filepath: str) -> str:
    target_path = validate_path(filepath, working_dir)
    if not target_path.exists(): return f"File {filepath} not found."
    with open(target_path, "r") as f: return f.read()

def list_files(working_dir: Path, directory: str) -> str:
    target_dir = validate_path(directory, working_dir)
    if not target_dir.exists() or not target_dir.is_dir(): return f"Not a directory: {directory}"
    files = [f"{item.relative_to(working_dir)}{'/' if item.is_dir() else ''}" for item in target_dir.iterdir()]
    return "\n".join(files) if files else "Directory is empty."

def parse_log(working_dir: Path, log_file: str, all_steps: bool, extract_performance: bool) -> str:
    log_path = validate_path(log_file, working_dir)
    if not log_path.exists(): return f"Log file {log_file} not found."
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
            output_parts.append("Thermodynamic Data:")
            for i, (head, block) in enumerate(data_blocks):
                output_parts.append(f"Run {i+1}: {', '.join(head)}")
                if all_steps:
                    for row in block: output_parts.append("  " + " ".join(row))
                elif block: output_parts.append(f"  Final: {' '.join(block[-1])}")
        if extract_performance:
            matches = re.findall(r"(Loop time of.*?)(?=\n\s*\n|Step|$)", content, re.DOTALL)
            if matches:
                output_parts.append("\nPerformance Data:")
                for i, m in enumerate(matches): output_parts.append(f"Run {i+1}:\n{m.strip()}")
        return "\n".join(output_parts) or "No data found."
    except Exception as e: return f"Error: {str(e)}"

async def serve(lammps_binary: str, working_directory: Path, remote: bool = False, host: str = "0.0.0.0", port: int = 8000) -> None:
    from .utils import get_system_info, get_lammps_capabilities
    server = Server("mcp-lammps")
    working_directory = working_directory.expanduser().resolve()
    working_directory.mkdir(parents=True, exist_ok=True)
    binary_cmd = shlex.split(lammps_binary, posix=(os.name != 'nt'))
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(name=LammpsTools.RUN, description="Run LAMMPS", inputSchema=LammpsRun.model_json_schema()),
            Tool(name=LammpsTools.READ_LOG, description="Read log", inputSchema=LammpsReadLog.model_json_schema()),
            Tool(name=LammpsTools.GET_THERMO, description="Get thermo", inputSchema=LammpsReadLog.model_json_schema()),
            Tool(name=LammpsTools.RESTART2DATA, description="Restart to data", inputSchema=LammpsRestart2Data.model_json_schema()),
            Tool(name=LammpsTools.UPLOAD_FILE, description="Upload file", inputSchema=LammpsUploadFile.model_json_schema()),
            Tool(name=LammpsTools.READ_FILE, description="Read file", inputSchema=LammpsReadFile.model_json_schema()),
            Tool(name=LammpsTools.LIST_FILES, description="List files", inputSchema=LammpsListFiles.model_json_schema()),
            Tool(name=LammpsTools.GET_INFO, description="Get info", inputSchema=LammpsGetInfo.model_json_schema()),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        match name:
            case LammpsTools.RUN:
                res = await run_lammps(binary_cmd, working_directory, LammpsRun(**arguments))
                return [TextContent(type="text", text=res)]
            case LammpsTools.READ_LOG | LammpsTools.GET_THERMO:
                args = LammpsReadLog(**arguments)
                return [TextContent(type="text", text=parse_log(working_directory, args.log_file, args.get_all_steps, args.extract_performance))]
            case LammpsTools.RESTART2DATA:
                args = LammpsRestart2Data(**arguments)
                res = await convert_restart2data(binary_cmd, working_directory, args.restart_file, args.data_file)
                return [TextContent(type="text", text=res)]
            case LammpsTools.UPLOAD_FILE:
                args = LammpsUploadFile(**arguments)
                return [TextContent(type="text", text=upload_file(working_directory, args.filepath, args.content))]
            case LammpsTools.READ_FILE:
                args = LammpsReadFile(**arguments)
                return [TextContent(type="text", text=read_file(working_directory, args.filepath))]
            case LammpsTools.LIST_FILES:
                args = LammpsListFiles(**arguments)
                return [TextContent(type="text", text=list_files(working_directory, args.directory))]
            case LammpsTools.GET_INFO:
                sys_info = get_system_info()
                lammps_caps = get_lammps_capabilities(binary_cmd[0])
                res = f"System: {sys_info}\nLAMMPS: Binary={binary_cmd}, Version={lammps_caps['version']}, Packages={lammps_caps['packages']}"
                return [TextContent(type="text", text=res)]
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
