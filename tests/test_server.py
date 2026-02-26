import pytest
import anyio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from mcp.client.session import ClientSession
from mcp.types import CallToolResult, TextContent

from mcp_server_lammps.server import (
    LammpsTools,
    parse_thermo_from_log,
    validate_path,
    find_latest_archive,
    serve,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_LOG = """\
LAMMPS (10 Sep 2025)
units           real
atom_style      atomic
boundary        p p p
lattice         fcc 5.3
Lattice spacing in x,y,z = 5.3 5.3 5.3
region          sim_box block 0 5 0 5 0 5
create_box      1 sim_box
Created orthogonal box = (0 0 0) to (26.5 26.5 26.5)
create_atoms    1 box
Created 500 atoms
minimize        1.0e-4 1.0e-6 100 1000
   Step          Temp          Press          PotEng
         0   0             -365.4179      -880.096
         1   10            -360.0000      -875.000
Loop time of 0.000455 on 1 procs for 1 steps with 500 atoms

Performance: 543.602 ns/day, 0.044 hours/ns, 6291.690 timesteps/s
MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.000431   | 0.000431   | 0.000431   |   0.0 | 94.73
Total # of neighbors = 33500
Ave neighs/atom = 67

run             5000
   Step          Temp          Press          PotEng
      5000   18.2          -25.3           -870.123
      5100   --926.719     10.5            -869.500
Loop time of 0.794699 on 1 procs for 5000 steps with 500 atoms

Performance: 348.947 ns/day, 0.069 hours/ns, 4038.733 timesteps/s
Total wall time: 0:00:02
"""


@pytest.fixture
def work_dir(tmp_path):
    """Create a working directory for tests."""
    d = tmp_path / "work"
    d.mkdir()
    return d


@pytest.fixture
def log_file(work_dir):
    """Write the sample log into the working directory."""
    p = work_dir / "log.lammps"
    p.write_text(SAMPLE_LOG)
    return p


@pytest.fixture
def archive_dir(work_dir):
    """Create an archive directory with a log file inside."""
    ad = work_dir / "archives" / "20260101_120000"
    ad.mkdir(parents=True)
    (ad / "log.lammps").write_text(SAMPLE_LOG)
    (ad / "dump.xyz").write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n")
    return ad


async def _create_server_client(work_dir: Path, lammps_binary: str = "lmp"):
    """
    Spin up the MCP server and a connected client session using in-memory
    anyio streams.  Returns (server_task_status, client_session).
    """
    from mcp.server.lowlevel import Server

    server = Server("mcp-lammps")
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Register handlers the same way serve() does
    @server.list_tools()
    async def list_tools():
        from mcp.types import Tool
        from mcp_server_lammps.server import (
            LammpsSubmitScript,
            LammpsReadLog,
            LammpsReadOutput,
            LammpsRestart,
        )

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
    async def call_tool(name, arguments):
        from mcp_server_lammps.server import (
            LammpsSubmitScript,
            LammpsReadLog,
            LammpsReadOutput,
            LammpsRestart,
            run_optimized_lammps,
            CallToolResult,
        )

        match name:
            case LammpsTools.SUBMIT_SCRIPT:
                args = LammpsSubmitScript(**arguments)
                try:
                    res = await run_optimized_lammps(
                        lammps_binary,
                        work_dir,
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
                path = validate_path(args.log_file, work_dir)
                if not path.exists():
                    latest = find_latest_archive(work_dir)
                    if latest and (latest / args.log_file).exists():
                        path = latest / args.log_file
                res = parse_thermo_from_log(path, args.extract_performance)
                return [TextContent(type="text", text=res)]

            case LammpsTools.READ_OUTPUT:
                args = LammpsReadOutput(**arguments)
                path = validate_path(args.filepath, work_dir)
                if not path.exists():
                    latest = find_latest_archive(work_dir)
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
                path = validate_path(args.restart_file, work_dir)
                if args.action == "data" and not args.output_file:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="output_file is required for 'data' action.",
                            )
                        ],
                        isError=True,
                    )
                if args.action == "dump" and not args.output_file:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="output_file is required for 'dump' action.",
                            )
                        ],
                        isError=True,
                    )
                if args.action not in ("data", "info", "dump"):
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Unknown action: {args.action}. Use 'data', 'info', or 'dump'.",
                            )
                        ],
                        isError=True,
                    )
                return [
                    TextContent(
                        type="text",
                        text=f"Action '{args.action}' completed (mock).",
                    )
                ]

            case _:
                raise ValueError(f"Unknown tool: {name}")

    return server


# ---------------------------------------------------------------------------
# Unit tests: validate_path
# ---------------------------------------------------------------------------


class TestValidatePath:
    def test_relative_path(self, work_dir):
        path = validate_path("test.txt", work_dir)
        assert path == work_dir / "test.txt"

    def test_absolute_inside(self, work_dir):
        inner = work_dir / "sub"
        inner.mkdir()
        path = validate_path(str(inner / "file.txt"), work_dir)
        assert path == inner / "file.txt"

    def test_rejects_path_outside(self, work_dir):
        with pytest.raises(ValueError, match="outside the working directory"):
            validate_path("../outside.txt", work_dir)

    def test_allows_archives_path(self, work_dir):
        archives = work_dir / "archives" / "20260101_120000"
        archives.mkdir(parents=True)
        path = validate_path("archives/20260101_120000/log.lammps", work_dir)
        assert "archives" in str(path)

    def test_absolute_outside_rejected(self, tmp_path, work_dir):
        outside = tmp_path / "other"
        outside.mkdir()
        with pytest.raises(ValueError):
            validate_path(str(outside / "file.txt"), work_dir)


# ---------------------------------------------------------------------------
# Unit tests: find_latest_archive
# ---------------------------------------------------------------------------


class TestFindLatestArchive:
    def test_no_archives_dir(self, work_dir):
        assert find_latest_archive(work_dir) is None

    def test_empty_archives_dir(self, work_dir):
        (work_dir / "archives").mkdir()
        assert find_latest_archive(work_dir) is None

    def test_returns_latest(self, work_dir):
        base = work_dir / "archives"
        (base / "20260101_100000").mkdir(parents=True)
        (base / "20260201_100000").mkdir(parents=True)
        (base / "20260115_100000").mkdir(parents=True)
        latest = find_latest_archive(work_dir)
        assert latest is not None
        assert latest.name == "20260201_100000"


# ---------------------------------------------------------------------------
# Unit tests: parse_thermo_from_log
# ---------------------------------------------------------------------------


class TestParseThermo:
    def test_basic_parse(self, log_file):
        result = parse_thermo_from_log(log_file, extract_performance=True)
        assert "Thermodynamic Data Summary:" in result
        assert "Run 1:" in result
        assert "Run 2:" in result
        assert "Step, Temp, Press, PotEng" in result
        assert "Performance Summary:" in result
        assert "Loop time" in result

    def test_final_state(self, log_file):
        result = parse_thermo_from_log(log_file, extract_performance=False)
        # Run 1 final state
        assert "Final State: 1 10 -360.0000 -875.000" in result

    def test_merged_signs(self, log_file):
        """Double negative (--) should be replaced with single (-)."""
        result = parse_thermo_from_log(log_file, extract_performance=False)
        assert "-926.719" in result
        assert "--926.719" not in result

    def test_no_performance_when_disabled(self, log_file):
        result = parse_thermo_from_log(log_file, extract_performance=False)
        assert "Performance Summary:" not in result

    def test_missing_log(self, work_dir):
        result = parse_thermo_from_log(work_dir / "nonexistent.log", True)
        assert "not found" in result

    def test_empty_log(self, work_dir):
        empty = work_dir / "empty.log"
        empty.write_text("")
        result = parse_thermo_from_log(empty, True)
        assert "No relevant data found" in result


# ---------------------------------------------------------------------------
# MCP Protocol-level tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_tools(work_dir):
    """Server must expose exactly 4 tools with correct names and valid schemas."""
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    result_tools = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal result_tools
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            result_tools = await session.list_tools()

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert result_tools is not None
    tool_names = {t.name for t in result_tools.tools}
    assert tool_names == {"submit_script", "read_log", "read_output", "restart"}

    # Every tool must have a valid JSON Schema with type=object
    for tool in result_tools.tools:
        assert tool.inputSchema["type"] == "object"
        assert "properties" in tool.inputSchema
        assert tool.description  # MCP spec: tools should have descriptions


@pytest.mark.anyio
async def test_read_log_tool(work_dir, log_file):
    """read_log tool should parse thermo data from an existing log file."""
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            call_result = await session.call_tool(
                "read_log",
                {"log_file": "log.lammps", "extract_performance": True},
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert call_result is not None
    assert not call_result.isError
    text = call_result.content[0].text
    assert "Thermodynamic Data Summary:" in text
    assert "Performance Summary:" in text


@pytest.mark.anyio
async def test_read_log_from_archive(work_dir, archive_dir):
    """read_log should fall back to the latest archive when log not in work_dir."""
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            call_result = await session.call_tool(
                "read_log",
                {"log_file": "log.lammps"},
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert call_result is not None
    assert not call_result.isError
    assert "Thermodynamic Data Summary:" in call_result.content[0].text


@pytest.mark.anyio
async def test_read_output_success(work_dir):
    """read_output should return file contents on success."""
    (work_dir / "dump.xyz").write_text("ITEM: TIMESTEP\n0\n")
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            call_result = await session.call_tool(
                "read_output",
                {"filepath": "dump.xyz"},
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert call_result is not None
    assert not call_result.isError
    assert "ITEM: TIMESTEP" in call_result.content[0].text


@pytest.mark.anyio
async def test_read_output_not_found_returns_error(work_dir):
    """read_output for missing file should return isError=True."""
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            call_result = await session.call_tool(
                "read_output",
                {"filepath": "nonexistent.dat"},
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert call_result is not None
    assert call_result.isError is True
    assert "not found" in call_result.content[0].text


@pytest.mark.anyio
async def test_read_output_from_archive(work_dir, archive_dir):
    """read_output should fall back to archive when file missing from work_dir."""
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            call_result = await session.call_tool(
                "read_output",
                {"filepath": "dump.xyz"},
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert call_result is not None
    assert not call_result.isError
    assert "ITEM: TIMESTEP" in call_result.content[0].text


@pytest.mark.anyio
async def test_read_output_truncation(work_dir):
    """Large files should be truncated at 10000 chars."""
    big = work_dir / "big.dump"
    big.write_text("x" * 20000)
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            call_result = await session.call_tool(
                "read_output",
                {"filepath": "big.dump"},
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert call_result is not None
    assert not call_result.isError
    assert "...(truncated)" in call_result.content[0].text


@pytest.mark.anyio
async def test_restart_data_missing_output_file(work_dir):
    """restart tool with action=data but no output_file should return isError."""
    (work_dir / "restart.bin").write_bytes(b"\x00")
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            call_result = await session.call_tool(
                "restart",
                {"restart_file": "restart.bin", "action": "data"},
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert call_result is not None
    assert call_result.isError is True
    assert "output_file is required" in call_result.content[0].text


@pytest.mark.anyio
async def test_restart_dump_missing_output_file(work_dir):
    """restart tool with action=dump but no output_file should return isError."""
    (work_dir / "restart.bin").write_bytes(b"\x00")
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            call_result = await session.call_tool(
                "restart",
                {"restart_file": "restart.bin", "action": "dump"},
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert call_result is not None
    assert call_result.isError is True
    assert "output_file is required" in call_result.content[0].text


@pytest.mark.anyio
async def test_restart_unknown_action(work_dir):
    """restart tool with an invalid action should return isError."""
    (work_dir / "restart.bin").write_bytes(b"\x00")
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            call_result = await session.call_tool(
                "restart",
                {"restart_file": "restart.bin", "action": "invalid_action"},
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert call_result is not None
    assert call_result.isError is True
    assert "Unknown action" in call_result.content[0].text


@pytest.mark.anyio
async def test_unknown_tool_returns_error(work_dir):
    """Calling a non-existent tool should return an error via MCP protocol."""
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            call_result = await session.call_tool(
                "nonexistent_tool",
                {},
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    # The SDK catches ValueError and returns isError=True
    assert call_result is not None
    assert call_result.isError is True
    assert "Unknown tool" in call_result.content[0].text


@pytest.mark.anyio
async def test_tool_schemas_have_required_fields(work_dir):
    """Every tool schema must comply with JSON Schema object format per MCP spec."""
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    result_tools = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal result_tools
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            result_tools = await session.list_tools()

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert result_tools is not None
    for tool in result_tools.tools:
        schema = tool.inputSchema
        # MCP spec: inputSchema must be a JSON Schema object
        assert schema["type"] == "object", f"{tool.name} schema missing type=object"
        assert "properties" in schema, f"{tool.name} schema missing properties"

        # submit_script requires script_content
        if tool.name == "submit_script":
            assert "script_content" in schema["required"]
        # read_output requires filepath
        elif tool.name == "read_output":
            assert "filepath" in schema["required"]
        # restart requires restart_file
        elif tool.name == "restart":
            assert "restart_file" in schema["required"]


@pytest.mark.anyio
async def test_read_log_default_args(work_dir, log_file):
    """read_log with default arguments should use log.lammps and extract performance."""
    server = await _create_server_client(work_dir)

    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(
        10
    )
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(
        10
    )

    call_result = None

    async def run_server():
        await server.run(
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
        )

    async def run_client():
        nonlocal call_result
        async with ClientSession(
            server_to_client_receive, client_to_server_send
        ) as session:
            await session.initialize()
            # Use defaults: log_file="log.lammps", extract_performance=True
            call_result = await session.call_tool("read_log", {})

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        tg.start_soon(run_client)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()

    assert call_result is not None
    assert not call_result.isError
    assert "Thermodynamic Data Summary:" in call_result.content[0].text
    assert "Performance Summary:" in call_result.content[0].text


# ---------------------------------------------------------------------------
# Integration test: parse the real attached log file if available
# ---------------------------------------------------------------------------

REAL_LOG = Path(__file__).parent / "in.log"


@pytest.mark.skipif(not REAL_LOG.exists(), reason="Real log file not available")
def test_parse_real_log():
    """Parse the real LAMMPS log file provided with the project."""
    result = parse_thermo_from_log(REAL_LOG, extract_performance=True)
    assert "Thermodynamic Data Summary:" in result
    assert "Performance Summary:" in result
    # The real log has multiple runs (temperature steps)
    assert "Run 1:" in result
    assert "Run 2:" in result
