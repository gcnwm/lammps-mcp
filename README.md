# mcp-server-lammps: A LAMMPS MCP server

## Overview

A Model Context Protocol server for LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator). This server allows Large Language Models to interact with LAMMPS to run simulations, validate scripts, manage files, and analyze output logs.

### Tools

1. `lammps_run`
   - Run a LAMMPS simulation with extensive configuration options.
   - Inputs:
     - `input_file`: Path to the input script (relative to working directory).
     - `log_file`: Output log file path (default: "log.lammps").
     - `screen_file`: Output screen file path (set to "none" to suppress).
     - `variables`: Dictionary of variables to define (e.g., `{"temp": "300.0"}` maps to `-var temp 300.0`).
     - `suffix`: Accelerator suffix style (e.g., "omp", "gpu").
     - `package`: Package command arguments (e.g., `["omp", "4"]`).
     - `options`: List of additional command-line flags (e.g., `["-partition", "8x2"]`).

2. `lammps_read_log`
   - Read thermodynamic and performance data from a LAMMPS log file.
   - Inputs:
     - `log_file`: Path to the log file.
     - `get_all_steps`: If true, returns data for all steps; otherwise returns only the final state of each run.
     - `extract_performance`: If true, extracts performance summaries (Loop time, MPI task breakdown, etc.).

3. `lammps_validate`
   - Validate a LAMMPS input script (syntax check only).
   - Inputs:
     - `input_file`: Path to the input script.

4. `lammps_restart2data`
   - Convert a binary restart file to a text data file.
   - Inputs:
     - `restart_file`: Path to the binary restart file.
     - `data_file`: Path to the output data file.

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
