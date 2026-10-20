#!/usr/bin/env python3
"""Emit source/lock/host/compiler provenance for retained witness diagnostics.

No adapter acquisition or compilation occurs here. Native shader compiler
versions that the host/driver does not expose remain unknown; Rust and locked
Naga identities are not substituted for those missing driver compiler versions.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess


def capture(command):
    """Retain optional host-tool failures instead of inventing an identity."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "unavailable": type(error).__name__}


def main():
    """Write one JSON identity record to the automatic workflow log."""
    toolchain = os.environ["RUST_TOOLCHAIN"]
    root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != root:
        raise RuntimeError("run qualification identity from the checkout root")
    head = capture(["git", "rev-parse", "HEAD", "HEAD^{tree}"])
    status = capture(["git", "status", "--porcelain=v1", "--untracked-files=normal"])
    rustc = capture(["rustc", "+" + toolchain, "-vV"])
    cargo = capture(["cargo", "+" + toolchain, "--version", "--verbose"])
    for required in (head, status, rustc, cargo):
        if required.get("exit_code") != 0:
            raise RuntimeError("required source/compiler identity unavailable: " + repr(required))
    manifests = [Path("Cargo.toml"), Path("Cargo.lock")]
    manifests.extend(sorted(Path("crates").glob("*/Cargo.toml")))
    lock = Path("Cargo.lock").read_text()
    shader_packages = [
        {"name": name, "version": version}
        for name, version in re.findall(r'name = "([^"]+)"\nversion = "([^"]+)"', lock)
        if name in {"wgpu", "wgpu-core", "wgpu-hal", "naga"}
    ]
    record = {
        "kind": "intrinsic-witness-cohort-provenance",
        "head_and_tree": head,
        "working_tree_status": status,
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in manifests},
        "locked_shader_packages": shader_packages,
        "runner_and_build": {
            name: os.environ[name]
            for name in (
                "ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH",
                "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA",
                "RUST_TOOLCHAIN", "RUSTFLAGS", "WGPU_BACKEND",
            )
            if name in os.environ
        },
        "rustc": rustc,
        "cargo": cargo,
        "os": platform.platform(),
        "uname": capture(["uname", "-a"]),
        "native_shader_compiler_version": "unknown unless independently exposed and bound to the run",
        "qualification": "pending actual six witness results, review and owner/coordinator acceptance",
    }
    if platform.system() == "Darwin":
        record["macos"] = capture(["sw_vers"])
        record["xcode"] = capture(["xcodebuild", "-version"])
    elif platform.system() == "Linux":
        record["os_release"] = Path("/etc/os-release").read_text()
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
