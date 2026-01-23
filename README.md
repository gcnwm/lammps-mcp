# mcp-server-lammps: A LAMMPS MCP server

## Overview

A Model Context Protocol server for LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator). This server allows Large Language Models to interact with LAMMPS to run simulations, validate scripts, and analyze output logs.

### Tools

1. `lammps_run`
   - Run a LAMMPS simulation.
   - Inputs:
     - `input_file`: Path to the input script (relative to working directory).
     - `log_file`: Output log file path (default: log.lammps).
     - `options`: List of additional command-line flags.

2. `lammps_read_log`
   - Read thermodynamic data from a LAMMPS log file.
   - Inputs:
     - `log_file`: Path to the log file.
     - `get_all_steps`: If true, returns all steps; otherwise returns only the final state.

<!-- 3. `lammps_validate`
   - Validate a LAMMPS input script (syntax check).
   - Inputs:
     - `input_file`: Path to the input script. -->

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

Make sure the `lammps-binary` path points to your LAMMPS executable (e.g., `lmp_serial`, `lmp_mpi`). The `working-directory` is where simulations will be run and where the server looks for files.
