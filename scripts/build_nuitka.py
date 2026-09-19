"""
Nuitka Native Build & Staging Verification Utility for Argus ANPR.

Compiles the first-party `app` package into native machine code (.so / .pyd)
without following or compiling imported third-party libraries (e.g. torch, ultralytics).

Usage:
    uv run python scripts/build_nuitka.py [--output-dir=dist] [--verify]
"""

import argparse
import os
import shutil
import subprocess
import sys

from loguru import logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the Nuitka build runner."""
    parser = argparse.ArgumentParser(description="Build Argus app package with Nuitka.")
    parser.add_argument(
        "--output-dir",
        default="compiled_dist",
        help="Directory where compiled extension module will be placed (default: compiled_dist).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Stage and verify that the compiled module imports cleanly without .py sources.",
    )
    return parser.parse_args()


def run_nuitka_build(output_dir: str) -> None:
    """
    Execute Nuitka to compile the app package into a native C-extension module.

    Enforces strict `--nofollow-imports` to ensure third-party wheels
    remain uncompiled and are loaded from the environment.
    """
    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--module",
        "--include-package=app",
        "--nofollow-imports",
        "--remove-output",
        "--lto=yes",
        "--python-flag=no_docstrings",
        f"--output-dir={output_dir}",
        "app",
    ]
    logger.info(f"Executing Nuitka build:\n  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def verify_compiled_module(output_dir: str) -> None:
    """
    Stage compiled artifacts in an isolated directory and verify importability.

    Confirms that Python can import the compiled app package without any
    .py source files present.
    """
    verify_dir = os.path.join(output_dir, "test_stage")
    os.makedirs(verify_dir, exist_ok=True)
    try:
        # Copy compiled .so/.pyd files to verification stage
        for item in os.listdir(output_dir):
            if item.endswith((".so", ".pyd", ".dylib")):
                shutil.copy2(os.path.join(output_dir, item), verify_dir)

        # Run python import check in the isolated directory
        check_cmd = [
            sys.executable,
            "-c",
            "import app; print('SUCCESS: Compiled app module loaded successfully!')",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(verify_dir)
        subprocess.run(check_cmd, check=True, cwd=verify_dir, env=env)
    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)


def main() -> None:
    """CLI orchestrator for Nuitka compilation and verification."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_nuitka_build(args.output_dir)

    if args.verify:
        logger.info("Verifying compiled module in isolated environment...")
        verify_compiled_module(args.output_dir)
        logger.info("Verification completed successfully.")


if __name__ == "__main__":
    main()
