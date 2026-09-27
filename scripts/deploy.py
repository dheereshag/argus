"""deploy.py — 1-command remote Pi deploy for Argus: compile, systemd, purge source."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

TARGET = "/opt/argus"
PORT = 8000

EXCLUDES = [
    "--exclude=.venv",
    "--exclude=.env*",
    "--exclude=__pycache__",
    "--exclude=.git",
    "--exclude=tests",
    "--exclude=docs",
    "--exclude=debug_crops",
    "--exclude=.cache",
    "--exclude=.coverage",
    "--exclude=.pytest_cache",
    "--exclude=.ruff_cache",
    "--exclude=compiled_dist",
    "--exclude=dist",
]


def log(msg: str) -> None:
    sys.stdout.write(f"{msg}\n")
    sys.stdout.flush()


def run_cmd(cmd: list[str]) -> None:
    if (code := subprocess.run(cmd, check=False).returncode) != 0:
        sys.stderr.write(f"[Deploy] Command failed: {' '.join(cmd)}\n")
        sys.exit(code)


def deploy_remote(host: str) -> None:
    user = host.split("@")[0] if "@" in host else os.environ.get("USER", "gluvok")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log(f"[Deploy] 1. Preparing {TARGET} on {host}...")
    prep = (
        f"sudo mkdir -p {TARGET} && sudo chown -R $USER:$USER {TARGET}; "
        f"command -v ccache >/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq ccache) || true; "
        f'export PATH="$HOME/.local/bin:$PATH"; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'
    )
    run_cmd(["ssh", "-t", host, prep])

    log(f"[Deploy] 2. Syncing source & model weights to {host}:{TARGET} (excluding .env)...")
    run_cmd(["rsync", "-avz", "--delete", *EXCLUDES, f"{repo_root}/", f"{host}:{TARGET}/"])

    log("[Deploy] 3. Compiling app module with Nuitka and purging source...")
    build = (
        f'set -e; export PATH="$HOME/.local/bin:$PATH"; cd {TARGET} && uv sync && '
        f"uv run python -m nuitka --module --include-package=app --nofollow-imports "
        f"--remove-output --lto=yes --python-flag=no_docstrings --output-dir={TARGET}/dist app && "
        f"mv -f {TARGET}/dist/app*so {TARGET}/ && "
        f"printf '#!/usr/bin/env bash\\nset -e\\ncd {TARGET}\\nexec {TARGET}/.venv/bin/fastapi run --host 127.0.0.1 --port {PORT}\\n' > {TARGET}/run.sh && "
        f"chmod +x {TARGET}/run.sh; "
        f"rm -rf {TARGET}/app {TARGET}/dist {TARGET}/tests {TARGET}/docs {TARGET}/debug_crops {TARGET}/scripts; "
        f"find {TARGET} -maxdepth 1 -name '*.py' -delete; "
        f"echo '[Deploy] Source purged! Only app.*.so and run.sh remain at {TARGET}.'"
    )
    run_cmd(["ssh", "-t", host, build])

    log("[Deploy] 4. Provisioning systemd service...")
    svc = (
        f"[Unit]\nDescription=Argus ANPR FastAPI Microservice\nAfter=network.target\n\n"
        f"[Service]\nType=simple\nUser={user}\nWorkingDirectory={TARGET}\n"
        f"ExecStart={TARGET}/run.sh\nRestart=always\nRestartSec=5\n"
        f"Environment=PYTHONUNBUFFERED=1\n\n[Install]\nWantedBy=multi-user.target\n"
    )
    dist_dir = os.path.join(repo_root, "dist")
    os.makedirs(dist_dir, exist_ok=True)
    svc_file = os.path.join(dist_dir, "argus.service")
    with open(svc_file, "w", encoding="utf-8") as f:
        f.write(svc)

    run_cmd(["scp", svc_file, f"{host}:/tmp/argus.service"])
    remote_svc_cmd = (
        "sudo mv /tmp/argus.service /etc/systemd/system/argus.service && "
        "sudo systemctl daemon-reload && sudo systemctl enable --now argus && "
        "sudo systemctl restart argus"
    )
    run_cmd(["ssh", "-t", host, remote_svc_cmd])
    log(f"\n[Deploy] Complete! Argus active & running 24/7 on {host} at 127.0.0.1:{PORT}.")


def main() -> None:
    p = argparse.ArgumentParser(description="Argus 1-Command Pi Deployer")
    p.add_argument("--host", default="gluvok@hermes.local", help="SSH host (e.g. gluvok@hermes.local)")
    a = p.parse_args()
    deploy_remote(a.host)


if __name__ == "__main__":
    main()
