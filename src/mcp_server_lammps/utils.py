import shutil
import os
import sys
import logging
import subprocess
import multiprocessing
import re
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """Get information about the system environment."""
    info = {
        "os": sys.platform,
        "cpu_cores": multiprocessing.cpu_count(),
        "mpi_available": shutil.which("mpirun") is not None or shutil.which("mpiexec") is not None,
    }
    return info

def get_lammps_capabilities(binary: str) -> Dict[str, Any]:
    """Get capabilities of the LAMMPS binary by running lmp -h."""
    caps = {
        "version": "unknown",
        "packages": [],
    }
    try:
        # Check if the binary exists or is in PATH
        if not Path(binary).exists() and not shutil.which(binary):
            return caps

        result = subprocess.run([binary, "-h"], capture_output=True, text=True, timeout=5)
        output = result.stdout

        # Parse version
        version_match = re.search(r"LAMMPS \((.*?)\)", output)
        if version_match:
            caps["version"] = version_match.group(1)

        # Parse packages
        if "Installed packages:" in output:
            package_section = output.split("Installed packages:")[1].split("\n\n")[0]
            caps["packages"] = [p.strip() for p in package_section.split() if p.strip()]

    except Exception as e:
        logger.error(f"Error getting LAMMPS capabilities: {e}")

    return caps

def find_lammps_binary() -> Optional[str]:
    """
    Search for LAMMPS binary in common locations.
    """
    # 1. Search in PATH
    for name in ["lmp", "lmp_serial", "lmp_mpi"]:
        binary = shutil.which(name)
        if binary:
            return binary

    # 2. Common Linux locations
    if sys.platform.startswith("linux"):
        linux_paths = ["/usr/local/bin/lmp", "/usr/bin/lmp", "/opt/lammps/bin/lmp"]
        for path_str in linux_paths:
            path = Path(path_str)
            if path.exists() and os.access(path, os.X_OK):
                return str(path)

    # 3. Common Windows locations
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        lammps_base = Path(program_files)
        if lammps_base.exists():
            for d in lammps_base.iterdir():
                if d.is_dir() and "LAMMPS" in d.name:
                    binary = d / "bin" / "lmp.exe"
                    if binary.exists():
                        return str(binary)

    return None
