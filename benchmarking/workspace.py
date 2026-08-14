import contextlib
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def run_cmd(cmd: list[str] | str, cwd: str | Path | None = None, check: bool = True, shell: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess command and stream output."""
    if isinstance(cmd, list):
        cmd_str = ' '.join(cmd)
    else:
        cmd_str = cmd
    
    result = subprocess.run(cmd, cwd=cwd, check=False, shell=shell)
    if check and result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}: {cmd_str}")
        sys.exit(result.returncode)
    return result

def check_vcs() -> str | None:
    """Detect if we are in a jj or git repo."""
    if subprocess.run(['jj', 'root'], capture_output=True).returncode == 0:
        return 'jj'
    if subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], capture_output=True).returncode == 0:
        return 'git'
    return None

def setup_baseline(baseline_dir: str | Path, baseline_rev: str, vcs: str) -> None:
    if Path(baseline_dir).exists():
        logger.info(">>> Removing existing baseline temporary directory...")
        shutil.rmtree(baseline_dir)

    logger.info(f">>> Setting up baseline workspace ({baseline_rev}) at {baseline_dir}...")
    if vcs == 'jj':
        run_cmd(['jj', 'workspace', 'add', str(baseline_dir), '-r', baseline_rev])
    elif vcs == 'git':
        run_cmd(['git', 'worktree', 'add', str(baseline_dir), baseline_rev])
    else:
        logger.error("Error: Neither a jj nor git repository detected.")
        sys.exit(1)

def cleanup_baseline(baseline_dir: str | Path, vcs: str) -> None:
    logger.info(">>> Cleaning up baseline workspace...")
    if vcs == 'jj':
        subprocess.run(['jj', 'workspace', 'forget', Path(baseline_dir).name], capture_output=True)
    elif vcs == 'git':
        subprocess.run(['git', 'worktree', 'remove', '--force', str(baseline_dir)], capture_output=True)
    
    if Path(baseline_dir).exists():
        shutil.rmtree(baseline_dir)

@contextlib.contextmanager
def managed_baseline(baseline_dir: str | Path, baseline_rev: str, vcs: str):
    setup_baseline(baseline_dir, baseline_rev, vcs)
    try:
        yield
    finally:
        cleanup_baseline(baseline_dir, vcs)

def build_all(workspaces: list[str | Path], skip_build: bool) -> None:
    if skip_build:
        return
    for d in workspaces:
        logger.info(f">>> Building //src:tesseract in {d}...")
        run_cmd(['bazel', 'build', '//src:tesseract'], cwd=str(d))
