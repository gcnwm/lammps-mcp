import pytest
from pathlib import Path
from mcp_server_lammps.server import parse_thermo_from_log, validate_path

def test_validate_path(tmp_path):
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    path = validate_path("test.txt", working_dir)
    assert path == working_dir / "test.txt"
    with pytest.raises(ValueError):
        validate_path("../outside.txt", working_dir)

def test_parse_log(tmp_path):
    working_dir = tmp_path
    log_file = working_dir / "log.lammps"
    log_file.write_text("""
   Step          Temp          Press
         0   0             -365.4179
         1   10            -360.0000
Loop time of 0.000455 on 1 procs for 1 steps with 500 atoms
""")
    result = parse_thermo_from_log(log_file, extract_performance=True)
    assert "Thermodynamic Data Summary:" in result
    assert "Run 1: Step, Temp, Press" in result
    assert "Final State: 1 10 -360.0000" in result
    assert "Performance Summary:" in result
    assert "Loop time of 0.000455" in result

def test_parse_log_with_merged_signs(tmp_path):
    working_dir = tmp_path
    log_file = working_dir / "log.lammps"
    log_file.write_text("""
   Step          PotEng
      4100   --926.719
Loop time of 0.1
""")
    result = parse_thermo_from_log(log_file, extract_performance=False)
    assert "-926.719" in result
