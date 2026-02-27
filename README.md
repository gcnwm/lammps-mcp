# mcp-server-lammps: A LAMMPS MCP server

## Overview

A Model Context Protocol server for LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator). This server allows Large Language Models to interact with LAMMPS to run simulations, validate scripts, manage files, and analyze output logs.

### Tools

1. `submit_script`
   - Submit and execute a LAMMPS input script.
   - Inputs:
     - `script_content`: Full LAMMPS input script text.
     - `script_name`: Saved input script filename (default: `in.lammps`).
     - `log_file`: Log filename (default: `log.lammps`).

2. `read_log`
   - Extract thermodynamic and timing/performance data from a log file.
   - Inputs:
     - `log_file`: Path to log (default: `log.lammps`).
     - `extract_performance`: Include loop/performance summary (default: `true`).

3. `read_output`
   - Read output file content (data/dump/custom text output).
   - Inputs:
     - `filepath`: Path to output file.

4. `restart`
   - Manage binary restart files.
   - Inputs:
     - `restart_file`: Path to restart file.
     - `action`: `data`, `info`, or `dump`.
     - `output_file`: Required for `data` and `dump`.

## Debug script

A fast smoke-test script is provided at `tests/debug_mcp_server.lmp`.

Typical workflow:

1. Start the server with a writable `--working-directory`.
2. Call `submit_script` with the contents of `tests/debug_mcp_server.lmp`.
3. Call `read_log` for `log.lammps`.
4. Call `read_output` for `debug_out/state.data`.
5. Call `restart` with `restart_file=debug_out/state.restart` and `action=info`.

## Installation

### Using uv (Recommended)

```bash
uvx mcp-server-lammps
```

### Using pip

```bash
pip install mcp-server-lammps
python -m mcp_server_lammps
```

## Configuration

### Usage with Claude Desktop

Add this to your `claude_desktop_config.json`:

```json
"mcpServers": {
  "lammps": {
    "command": "uvx",
    "args": [
      "mcp-server-lammps",
      "--lammps-binary", "/path/to/lmp",
      "--working-directory", "/path/to/simulations"
    ]
  }
}
```

### Windows Configuration

On Windows, ensure you escape backslashes in paths correctly (use double backslashes):

```json
"mcpServers": {
  "lammps": {
    "command": "uvx",
    "args": [
      "mcp-server-lammps",
      "--lammps-binary", "C:\\Program Files\\LAMMPS\\bin\\lmp.exe",
      "--working-directory", "D:\\Simulations\\WorkDir"
    ]
  }
}
```

Make sure the `lammps-binary` path points to your LAMMPS executable (e.g., `lmp_serial`, `lmp_mpi`). You can also specify an MPI command string if you want to run in parallel by default, e.g., `mpiexec -np 4 lmp_mpi`.

The `working-directory` is where simulations will be run and where the server looks for files. All file paths provided to tools must be within this directory.

## License

MIT
