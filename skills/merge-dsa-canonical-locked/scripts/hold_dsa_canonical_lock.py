#!/usr/bin/env python3
"""Hold a host-wide, per-target lock for canonical DSA promotion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
from typing import NoReturn


CANONICAL_TARGETS = {"v0", "v1"}
DEFAULT_LOCK_DIR = Path("/tmp/cudnn-frontend-dsa-canonical-locks")
RELEASE_WORD = "release"


class SignalExit(Exception):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def parse_args() -> argparse.Namespace:
    raw_args = sys.argv[1:]
    if "--" in raw_args:
        separator = raw_args.index("--")
        wrapper_args = raw_args[:separator]
        command = raw_args[separator + 1 :]
    else:
        wrapper_args = raw_args
        command = []

    parser = argparse.ArgumentParser(
        description=(
            "Hold an exclusive host-wide lock while promoting into canonical "
            "DSA v0 or v1."
        )
    )
    parser.add_argument(
        "target",
        choices=sorted(CANONICAL_TARGETS),
        help="canonical implementation token",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="hold until the exact line 'release' is received on stdin",
    )
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=DEFAULT_LOCK_DIR,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(wrapper_args)
    if args.hold == bool(command):
        parser.error("choose exactly one of --hold or -- command [args...]")
    args.command = command
    return args


def signal_exit(signum: int, _frame: object) -> NoReturn:
    raise SignalExit(signum)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, signal_exit)
    signal.signal(signal.SIGTERM, signal_exit)


def acquire_lock(lock_path: Path, target: str) -> int:
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    print(
        f"DSA_CANONICAL_LOCK_WAIT target={target} path={lock_path}",
        flush=True,
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    except BaseException:
        os.close(lock_fd)
        raise
    return lock_fd


def write_owner(lock_fd: int, target: str) -> None:
    owner = {
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "target": target,
    }
    payload = (json.dumps(owner, sort_keys=True) + "\n").encode()
    os.ftruncate(lock_fd, 0)
    os.lseek(lock_fd, 0, os.SEEK_SET)
    os.write(lock_fd, payload)
    os.fsync(lock_fd)


def hold_until_release() -> int:
    print(
        "DSA_CANONICAL_LOCK_PROTOCOL send_exact_line=release",
        flush=True,
    )
    while True:
        line = sys.stdin.readline()
        if line == "":
            print("DSA_CANONICAL_LOCK_STDIN_EOF", flush=True)
            return 0
        if line.rstrip("\r\n") == RELEASE_WORD:
            return 0
        print("DSA_CANONICAL_LOCK_INPUT_IGNORED", flush=True)


def wait_for_child(child: subprocess.Popen[bytes]) -> int:
    try:
        return child.wait()
    except SignalExit as exc:
        if child.poll() is None:
            child.send_signal(exc.signum)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        return 128 + exc.signum
    except KeyboardInterrupt:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        return 128 + signal.SIGINT


def run_command(command: list[str]) -> int:
    print(
        f"DSA_CANONICAL_LOCK_COMMAND {shlex.join(command)}",
        flush=True,
    )
    try:
        child = subprocess.Popen(command)
    except FileNotFoundError:
        print(
            f"ERROR: command not found: {command[0]}",
            file=sys.stderr,
        )
        return 127
    return wait_for_child(child)


def clear_and_release(lock_fd: int, lock_path: Path, target: str) -> None:
    os.ftruncate(lock_fd, 0)
    os.fsync(lock_fd)
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)
    print(
        f"DSA_CANONICAL_LOCK_RELEASED target={target} path={lock_path}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    install_signal_handlers()
    lock_path = args.lock_dir.resolve() / f"{args.target}.lock"
    lock_fd: int | None = None
    try:
        lock_fd = acquire_lock(lock_path, args.target)
        write_owner(lock_fd, args.target)
        print(
            "DSA_CANONICAL_LOCK_ACQUIRED "
            f"target={args.target} path={lock_path} pid={os.getpid()}",
            flush=True,
        )
        if args.hold:
            return hold_until_release()
        return run_command(args.command)
    except SignalExit as exc:
        return 128 + exc.signum
    finally:
        if lock_fd is not None:
            clear_and_release(lock_fd, lock_path, args.target)


if __name__ == "__main__":
    raise SystemExit(main())
