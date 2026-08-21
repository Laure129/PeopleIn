"""Remove competing workloads before a benchmark run."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
import signal
import time


AI_PROCESS_NAMES = {"codex", "codex-code-mode", "kimi-code"}
CPU_LIMIT_PERCENT = 10.0
SAMPLE_SECONDS = 1.0
TERM_GRACE_SECONDS = 2.0


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    sid: int
    uid: int
    state: str
    comm: str
    cpu_ticks: int
    start_ticks: int


def snapshot():
    processes = {}
    for root in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(root.name)
            raw = (root / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2:].split()
            processes[pid] = Process(
                pid=pid,
                ppid=int(fields[1]),
                sid=int(fields[3]),
                uid=root.stat().st_uid,
                state=fields[0],
                comm=(root / "comm").read_text(encoding="utf-8").strip(),
                cpu_ticks=int(fields[11]) + int(fields[12]),
                start_ticks=int(fields[19]),
            )
        except (OSError, IndexError, ValueError):
            continue
    return processes


def ancestor_sids(processes, pid):
    protected = set()
    while pid in processes:
        process = processes[pid]
        protected.add(process.sid)
        if process.ppid == pid:
            break
        pid = process.ppid
    return protected


def select_targets(before, after, protected_sids, own_uid, elapsed, clock_ticks):
    ai_sids = {
        process.sid
        for process in after.values()
        if process.comm in AI_PROCESS_NAMES and process.sid not in protected_sids
    }
    targets = {
        process.pid for process in after.values() if process.sid in ai_sids
    }
    for pid, process in after.items():
        previous = before.get(pid)
        if previous is None or process.sid in protected_sids:
            continue
        cpu_percent = (
            (process.cpu_ticks - previous.cpu_ticks)
            / clock_ticks / elapsed * 100.0
        )
        if cpu_percent >= CPU_LIMIT_PERCENT:
            targets.add(pid)
    blocked = {pid for pid in targets if after[pid].uid != own_uid}
    return targets - blocked, blocked


def terminate(processes, pids):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        current = snapshot()
        alive = []
        for pid in sorted(pids, reverse=True):
            if (
                pid not in current
                or current[pid].start_ticks != processes[pid].start_ticks
            ):
                continue
            try:
                os.kill(pid, sig)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            break
        deadline = time.monotonic() + TERM_GRACE_SECONDS
        while time.monotonic() < deadline:
            remaining = snapshot()
            alive = [
                pid for pid in alive
                if pid in remaining and remaining[pid].state != "Z"
            ]
            if not alive:
                return
            time.sleep(0.1)
        pids = alive
    remaining = snapshot()
    survivors = [
        pid for pid in pids
        if (
            pid in remaining
            and remaining[pid].state != "Z"
            and remaining[pid].start_ticks == processes[pid].start_ticks
        )
    ]
    if survivors:
        raise RuntimeError(f"could not stop processes: {survivors}")


def describe(processes, pids):
    return ", ".join(
        f"{pid}:{processes[pid].comm}" for pid in sorted(pids)
    )


def prepare_benchmark():
    before = snapshot()
    protected_sids = ancestor_sids(before, os.getpid())
    started = time.monotonic()
    time.sleep(SAMPLE_SECONDS)
    after = snapshot()
    targets, blocked = select_targets(
        before,
        after,
        protected_sids,
        os.getuid(),
        time.monotonic() - started,
        os.sysconf("SC_CLK_TCK"),
    )
    if blocked:
        raise RuntimeError(
            "benchmark blocked by processes owned by another user: "
            + describe(after, blocked)
        )
    if targets:
        logging.info(
            "terminating benchmark competitors: %s",
            describe(after, targets),
        )
        terminate(after, targets)
    else:
        logging.info("benchmark host is clear")
