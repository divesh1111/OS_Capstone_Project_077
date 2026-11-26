"""OS Resource Studio — Unified Single-File Academic Simulator & Visualizer."""
from __future__ import annotations

import json
import re
import sys
import unittest
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

# Direct-launch & CLI test support before Streamlit UI initialization
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--test"]
        # Test classes are defined below; defer execution until end of module
        _RUN_TESTS_ON_MAIN = True
    else:
        _RUN_TESTS_ON_MAIN = False
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            from streamlit.web import cli as stcli

            sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
            sys.exit(stcli.main())
else:
    _RUN_TESTS_ON_MAIN = False

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


# ============================================================================
# CORE 1: CPU SCHEDULING ENGINE
# ============================================================================
@dataclass(frozen=True)
class Process:
    pid: str
    arrival: int
    burst: int
    priority: int = 0

    def __post_init__(self) -> None:
        if not str(self.pid).strip():
            raise ValueError("Process ID cannot be empty.")
        if self.arrival < 0:
            raise ValueError(f"Arrival time for {self.pid} must be non-negative.")
        if self.burst <= 0:
            raise ValueError(f"Burst time for {self.pid} must be positive.")


def _build_segments_with_idle(timeline: list[tuple[int, int, str]]) -> list[dict]:
    """Convert (start, end, pid) tuples into Gantt segments, inserting IDLE gaps."""
    segments: list[dict] = []
    cursor = 0
    for start, end, pid in timeline:
        if start > cursor:
            segments.append({"pid": "IDLE", "start": cursor, "end": start})
        segments.append({"pid": pid, "start": start, "end": end})
        cursor = end
    return segments


def _cpu_metrics(
    processes: list[Process],
    completion: dict[str, int],
    timeline: list[tuple[int, int, str]],
    algorithm: str,
) -> dict:
    rows = []
    for p in processes:
        turnaround = completion[p.pid] - p.arrival
        waiting = turnaround - p.burst
        response = next(s for s, _, pid in timeline if pid == p.pid) - p.arrival
        rows.append(
            {
                "pid": p.pid,
                "arrival": p.arrival,
                "burst": p.burst,
                "priority": p.priority,
                "completion": completion[p.pid],
                "turnaround": turnaround,
                "waiting": waiting,
                "response": response,
            }
        )

    makespan = max(completion.values()) if completion else 0
    total_burst = sum(p.burst for p in processes)
    cpu_utilization = (total_burst / makespan * 100.0) if makespan > 0 else 0.0
    throughput = (len(processes) / makespan) if makespan > 0 else 0.0
    segments = _build_segments_with_idle(timeline)
    non_idle = [s for s in segments if s["pid"] != "IDLE"]
    context_switches = max(0, len(non_idle) - 1)

    return {
        "algorithm": algorithm,
        "processes": rows,
        "timeline": timeline,
        "segments": segments,
        "avg_waiting": sum(r["waiting"] for r in rows) / len(rows),
        "avg_turnaround": sum(r["turnaround"] for r in rows) / len(rows),
        "avg_response": sum(r["response"] for r in rows) / len(rows),
        "makespan": makespan,
        "cpu_utilization": cpu_utilization,
        "throughput": throughput,
        "context_switches": context_switches,
    }


def cpu_schedule(processes: list[Process], algorithm: str = "fcfs", quantum: int = 2) -> dict:
    """Run FCFS, SJF, Priority (lower = higher priority), or Round Robin."""
    ps = list(processes)
    if not ps:
        raise ValueError("At least one process is required.")
    if len({p.pid for p in ps}) != len(ps):
        raise ValueError("Process IDs must be unique.")

    alg = algorithm.lower()
    if alg not in ("fcfs", "sjf", "priority", "rr"):
        raise ValueError(f"Unknown scheduling algorithm: {algorithm}")
    if alg == "rr" and quantum <= 0:
        raise ValueError("Round-robin quantum must be positive.")

    now = 0
    completion: dict[str, int] = {}
    timeline: list[tuple[int, int, str]] = []

    def run(p: Process, duration: int) -> None:
        nonlocal now
        start = now
        now += duration
        timeline.append((start, now, p.pid))

    if alg in ("fcfs", "sjf", "priority"):
        pending = ps[:]
        while pending:
            ready = [p for p in pending if p.arrival <= now]
            if not ready:
                now = min(p.arrival for p in pending)
                ready = [p for p in pending if p.arrival <= now]
            if alg == "fcfs":
                p = min(ready, key=lambda x: (x.arrival, ps.index(x)))
            elif alg == "sjf":
                p = min(ready, key=lambda x: (x.burst, x.arrival, ps.index(x)))
            else:
                p = min(ready, key=lambda x: (x.priority, x.arrival, ps.index(x)))
            run(p, p.burst)
            completion[p.pid] = now
            pending.remove(p)
    else:
        arrivals = sorted(ps, key=lambda p: (p.arrival, ps.index(p)))
        remaining = {p.pid: p.burst for p in ps}
        ready: deque[Process] = deque()
        i = 0
        while i < len(arrivals) or ready:
            if not ready and i < len(arrivals) and now < arrivals[i].arrival:
                now = arrivals[i].arrival
            while i < len(arrivals) and arrivals[i].arrival <= now:
                ready.append(arrivals[i])
                i += 1
            p = ready.popleft()
            duration = min(quantum, remaining[p.pid])
            run(p, duration)
            remaining[p.pid] -= duration
            while i < len(arrivals) and arrivals[i].arrival <= now:
                ready.append(arrivals[i])
                i += 1
            if remaining[p.pid] > 0:
                ready.append(p)
            else:
                completion[p.pid] = now

    return _cpu_metrics(ps, completion, timeline, alg)


# ============================================================================
# CORE 2: CONTIGUOUS MEMORY ALLOCATION ENGINE
# ============================================================================
def mem_allocate(blocks: list[int], requests: list[int], strategy: str = "first_fit") -> dict:
    """Allocate contiguous memory requests across fixed blocks."""
    if not blocks or any(not isinstance(x, int) or x <= 0 for x in blocks):
        raise ValueError("Memory blocks must be positive integers.")
    if any(not isinstance(x, int) or x <= 0 for x in requests):
        raise ValueError("Memory requests must be positive integers.")

    strat = strategy.lower()
    if strat not in ("first_fit", "best_fit", "worst_fit"):
        raise ValueError(f"Unknown allocation strategy: {strategy}")

    free = list(blocks)
    block_contents: list[list[dict]] = [[] for _ in blocks]
    rows: list[dict] = []

    for idx, size in enumerate(requests, 1):
        candidates = [i for i, b in enumerate(free) if b >= size]
        chosen = None
        if candidates:
            if strat == "first_fit":
                chosen = candidates[0]
            elif strat == "best_fit":
                chosen = min(candidates, key=lambda i: free[i])
            else:
                chosen = max(candidates, key=lambda i: free[i])
            free[chosen] -= size
            block_contents[chosen].append({"request": f"R{idx}", "size": size})

        rows.append(
            {
                "request": idx,
                "label": f"R{idx}",
                "size": size,
                "block": chosen + 1 if chosen is not None else None,
                "remaining_in_block": free[chosen] if chosen is not None else None,
                "allocated": chosen is not None,
            }
        )

    total_capacity = sum(blocks)
    total_allocated = sum(r["size"] for r in rows if r["allocated"])
    total_free = sum(free)
    allocated_count = sum(1 for r in rows if r["allocated"])

    return {
        "strategy": strat,
        "allocations": rows,
        "initial_blocks": list(blocks),
        "final_free": free,
        "block_contents": block_contents,
        "allocated_count": allocated_count,
        "unallocated_count": len(rows) - allocated_count,
        "total_capacity": total_capacity,
        "total_allocated": total_allocated,
        "total_free": total_free,
        "utilization_pct": (total_allocated / total_capacity * 100.0) if total_capacity else 0.0,
        "internal_fragmentation": total_free,
    }


# ============================================================================
# CORE 3: BANKER'S DEADLOCK AVOIDANCE ENGINE
# ============================================================================
def banker_safety(available: list[int], allocation: list[list[int]], maximum: list[list[int]]) -> dict:
    """Evaluate system safety using Banker's algorithm and return execution trace."""
    n = len(allocation)
    m = len(available)
    if n == 0 or len(maximum) != n or any(len(r) != m for r in allocation + maximum):
        raise ValueError("Resource matrices have inconsistent dimensions.")
    if any(x < 0 for x in available) or any(x < 0 for row in allocation + maximum for x in row):
        raise ValueError("Resource counts cannot be negative.")
    if any(maximum[i][j] < allocation[i][j] for i in range(n) for j in range(m)):
        raise ValueError("Maximum claim cannot be below current allocation.")

    work = list(available)
    finish = [False] * n
    sequence: list[int] = []
    need = [[maximum[i][j] - allocation[i][j] for j in range(m)] for i in range(n)]
    trace: list[dict] = []

    while True:
        found = False
        for i in range(n):
            if not finish[i] and all(need[i][j] <= work[j] for j in range(m)):
                work_before = list(work)
                work = [work[j] + allocation[i][j] for j in range(m)]
                finish[i] = True
                sequence.append(i)
                trace.append(
                    {
                        "step": len(sequence),
                        "process": f"P{i}",
                        "need": list(need[i]),
                        "work_before": work_before,
                        "allocation": list(allocation[i]),
                        "work_after": list(work),
                    }
                )
                found = True
        if not found:
            break

    is_safe = all(finish)
    return {
        "safe": is_safe,
        "sequence": sequence if is_safe else [],
        "unfinished": [i for i, f in enumerate(finish) if not f],
        "need": need,
        "trace": trace,
    }


def request_resources(
    available: list[int],
    allocation: list[list[int]],
    maximum: list[list[int]],
    process: int,
    request: list[int],
) -> dict:
    """Validate and tentatively grant a resource request if the resulting state is safe."""
    if not 0 <= process < len(allocation):
        raise ValueError("Process index out of range.")
    if len(request) != len(available) or any(x < 0 for x in request):
        raise ValueError("Invalid request vector.")

    need = [maximum[process][j] - allocation[process][j] for j in range(len(available))]
    if any(request[j] > need[j] for j in range(len(request))):
        return {"granted": False, "reason": "Request exceeds declared remaining need."}
    if any(request[j] > available[j] for j in range(len(request))):
        return {"granted": False, "reason": "Requested resources exceed currently available units."}

    next_alloc = [row[:] for row in allocation]
    next_avail = list(available)
    for j, x in enumerate(request):
        next_alloc[process][j] += x
        next_avail[j] -= x

    check = banker_safety(next_avail, next_alloc, maximum)
    return {
        "granted": check["safe"],
        "reason": "Safe state preserved" if check["safe"] else "Request would leave the system in an unsafe state",
        "safety": check,
        "new_available": next_avail,
        "new_allocation": next_alloc,
    }


# ============================================================================
# CORE 4: DISK HEAD SCHEDULING ENGINE
# ============================================================================
def disk_schedule(
    requests: list[int],
    head: int,
    algorithm: str = "fcfs",
    disk_size: int = 200,
    direction: str = "right",
) -> dict:
    """Simulate disk head movement and return service order, path, and seek metrics."""
    if not requests:
        raise ValueError("At least one disk track request is required.")
    if disk_size <= 0 or not 0 <= head < disk_size or any(not 0 <= x < disk_size for x in requests):
        raise ValueError(f"Track positions must be within [0, {disk_size - 1}].")

    alg = algorithm.lower()
    dir_norm = direction.lower()
    if alg not in ("fcfs", "sstf", "scan"):
        raise ValueError(f"Unknown disk algorithm: {algorithm}")
    if dir_norm not in ("left", "right"):
        raise ValueError("SCAN direction must be 'left' or 'right'.")

    remaining = list(enumerate(requests))
    order: list[int] = []
    path: list[int] = [head]
    current = head

    if alg == "fcfs":
        order = [x for _, x in remaining]
    elif alg == "sstf":
        while remaining:
            idx, x = min(remaining, key=lambda pair: (abs(pair[1] - current), pair[0]))
            order.append(x)
            current = x
            remaining.remove((idx, x))
    else:
        left = sorted((x for x in requests if x < head), reverse=True)
        right = sorted(x for x in requests if x >= head)
        if dir_norm == "right":
            order = right + ([disk_size - 1] if left else []) + left
        else:
            order = left + ([0] if right else []) + right

    step_distances: list[dict] = []
    total = 0
    for step_idx, target in enumerate(order, 1):
        prev = path[-1]
        dist = abs(target - prev)
        total += dist
        path.append(target)
        step_distances.append(
            {
                "step": step_idx,
                "from": prev,
                "to": target,
                "distance": dist,
                "cumulative": total,
            }
        )

    return {
        "algorithm": alg,
        "head": head,
        "direction": dir_norm,
        "order": order,
        "path": path,
        "step_distances": step_distances,
        "total_movement": total,
        "average_movement": total / len(requests),
        "disk_size": disk_size,
    }


# ============================================================================
# CORE 5: CURATED WORKLOAD PRESETS
# ============================================================================
PRESETS: dict[str, dict] = {
    "Lab Standard (Default)": {
        "quantum": 2,
        "processes": [
            {"pid": "P1", "arrival": 0, "burst": 7, "priority": 2},
            {"pid": "P2", "arrival": 2, "burst": 4, "priority": 1},
            {"pid": "P3", "arrival": 4, "burst": 1, "priority": 3},
            {"pid": "P4", "arrival": 5, "burst": 4, "priority": 2},
        ],
        "memory": {
            "blocks": [100, 500, 200, 300, 600],
            "requests": [212, 417, 112, 426, 95],
        },
        "deadlock": {
            "available": [3, 3, 2],
            "allocation": [
                [0, 1, 0],
                [2, 0, 0],
                [3, 0, 2],
                [2, 1, 1],
                [0, 0, 2],
            ],
            "maximum": [
                [7, 5, 3],
                [3, 2, 2],
                [9, 0, 2],
                [2, 2, 2],
                [4, 3, 3],
            ],
        },
        "disk": {
            "requests": [98, 183, 37, 122, 14, 124, 65, 67],
            "head": 53,
            "disk_size": 200,
            "direction": "right",
        },
    },
    "Idle Gaps & Convoy": {
        "quantum": 3,
        "processes": [
            {"pid": "P1", "arrival": 1, "burst": 9, "priority": 3},
            {"pid": "P2", "arrival": 2, "burst": 2, "priority": 1},
            {"pid": "P3", "arrival": 3, "burst": 1, "priority": 2},
            {"pid": "P4", "arrival": 16, "burst": 4, "priority": 1},
            {"pid": "P5", "arrival": 17, "burst": 3, "priority": 2},
        ],
        "memory": {
            "blocks": [150, 350, 250, 450, 180],
            "requests": [140, 200, 310, 175, 90],
        },
        "deadlock": {
            "available": [2, 3, 0],
            "allocation": [
                [0, 1, 0],
                [2, 0, 0],
                [3, 0, 3],
                [2, 1, 1],
                [0, 0, 2],
            ],
            "maximum": [
                [7, 5, 3],
                [3, 2, 2],
                [9, 0, 3],
                [2, 2, 2],
                [4, 3, 3],
            ],
        },
        "disk": {
            "requests": [176, 79, 34, 60, 92, 11, 41, 114],
            "head": 50,
            "disk_size": 200,
            "direction": "left",
        },
    },
    "Unsafe Deadlock & Fragmentation": {
        "quantum": 2,
        "processes": [
            {"pid": "P1", "arrival": 0, "burst": 6, "priority": 2},
            {"pid": "P2", "arrival": 0, "burst": 3, "priority": 1},
            {"pid": "P3", "arrival": 1, "burst": 5, "priority": 3},
            {"pid": "P4", "arrival": 2, "burst": 2, "priority": 1},
        ],
        "memory": {
            "blocks": [200, 150, 300, 120],
            "requests": [180, 140, 250, 130, 110],
        },
        "deadlock": {
            "available": [1, 0, 0],
            "allocation": [
                [1, 1, 0],
                [2, 1, 1],
                [1, 0, 2],
                [0, 2, 1],
            ],
            "maximum": [
                [4, 2, 2],
                [3, 3, 2],
                [3, 1, 3],
                [2, 3, 2],
            ],
        },
        "disk": {
            "requests": [190, 10, 175, 25, 160, 40, 145],
            "head": 100,
            "disk_size": 200,
            "direction": "right",
        },
    },
}


def get_preset(name: str) -> dict:
    """Return a deep copy of the named preset."""
    return deepcopy(PRESETS[name])


# ============================================================================
# UTILS: ACADEMIC PLOTLY CHARTS
# ============================================================================
PALETTE = [
    "#1F4E79",
    "#2E75B6",
    "#A6611A",
    "#4D7C5F",
    "#7F5A83",
    "#B08D57",
    "#6B7B8C",
    "#9C4A3C",
]
INK = "#1A1A1A"
MUTED = "#5F5F5F"
FAINT = "#8A8A8A"
RULE = "#DAD5C9"
GRID = "#E7E2D8"
IDLE = "#B9B2A3"
FREE_FILL = "#EFECE4"
SURFACE = "#FFFFFF"
SERIF = "Source Serif 4, Georgia, serif"
MONO = "IBM Plex Mono, SFMono-Regular, Consolas, monospace"


def _pid_sort_key(pid: str) -> tuple:
    prefix, digits = re.match(r"^(\D*)(\d*)", str(pid)).groups()
    return (prefix, int(digits) if digits else -1, str(pid))


def style_fig(fig: go.Figure, title: str = "", height: int = 320) -> go.Figure:
    """Apply academic paper styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=height,
        font=dict(family=MONO, size=11, color=MUTED),
        title=dict(
            text=title,
            font=dict(family=SERIF, size=15, color=INK),
            x=0,
            xanchor="left",
        )
        if title
        else None,
        margin=dict(l=48, r=20, t=48 if title else 20, b=44),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED), title=None),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=RULE, font=dict(family=MONO, size=11, color=INK)),
        xaxis=dict(
            gridcolor=GRID,
            linecolor=IDLE,
            linewidth=1,
            zeroline=False,
            tickfont=dict(family=MONO, size=10, color=FAINT),
        ),
        yaxis=dict(
            gridcolor=GRID,
            linecolor=IDLE,
            linewidth=1,
            zeroline=False,
            tickfont=dict(family=MONO, size=10, color=FAINT),
        ),
    )
    return fig


def make_gantt(segments: list[dict]) -> go.Figure:
    """Single-row horizontal CPU execution trace with idle gaps."""
    pids = sorted(
        (p for p in dict.fromkeys(s["pid"] for s in segments) if p != "IDLE"),
        key=_pid_sort_key,
    )
    colors = {"IDLE": IDLE}
    colors.update({p: PALETTE[i % len(PALETTE)] for i, p in enumerate(pids)})

    fig = go.Figure()
    for s in segments:
        dur = s["end"] - s["start"]
        fig.add_trace(
            go.Bar(
                x=[dur],
                y=["CPU"],
                base=[s["start"]],
                orientation="h",
                name=s["pid"],
                marker_color=colors[s["pid"]],
                marker_line_color=SURFACE,
                marker_line_width=1,
                showlegend=False,
                text=[s["pid"]],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(family=MONO, size=10, color="#FFFFFF"),
                hovertemplate=(
                    f"<b>{s['pid']}</b><br>"
                    f"start&nbsp;&nbsp;{s['start']}<br>"
                    f"end&nbsp;&nbsp;&nbsp;&nbsp;{s['end']}<br>"
                    f"duration&nbsp;{dur}"
                    "<extra></extra>"
                ),
            )
        )

    end = max((s["end"] for s in segments), default=1)
    fig.update_layout(
        barmode="overlay",
        height=195,
        margin=dict(l=18, r=18, t=16, b=40),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=MONO, size=11, color=MUTED),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=RULE, font=dict(family=MONO, size=11, color=INK)),
        xaxis=dict(
            title=dict(text="time (units)", font=dict(family=MONO, size=10, color=FAINT)),
            range=[0, end],
            dtick=1 if end <= 30 else None,
            gridcolor=GRID,
            gridwidth=1,
            linecolor=IDLE,
            linewidth=1,
            zeroline=False,
            tickfont=dict(family=MONO, size=10, color=FAINT),
        ),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    return fig


def make_cpu_comparison_chart(cpu_results: dict[str, dict]) -> go.Figure:
    """Grouped bar chart comparing Avg Waiting, Turnaround, and Response across policies."""
    algs = [a.upper() for a in cpu_results.keys()]
    metrics = [
        ("Avg Waiting", "avg_waiting", PALETTE[0]),
        ("Avg Turnaround", "avg_turnaround", PALETTE[1]),
        ("Avg Response", "avg_response", PALETTE[3]),
    ]
    fig = go.Figure()
    for label, key, color in metrics:
        vals = [round(cpu_results[a.lower()][key], 2) for a in algs]
        fig.add_trace(
            go.Bar(
                name=label,
                x=algs,
                y=vals,
                marker_color=color,
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
                textfont=dict(family=MONO, size=10, color=MUTED),
            )
        )
    style_fig(fig, title="Scheduling Policy Comparison (time units)", height=330)
    fig.update_layout(barmode="group", yaxis_title="time (units)")
    return fig


def make_memory_map(mem_result: dict) -> go.Figure:
    """Horizontal stacked bar chart showing each block's allocated requests and free space."""
    blocks = mem_result["initial_blocks"]
    contents = mem_result["block_contents"]
    final_free = mem_result["final_free"]
    block_labels = [f"Block {i + 1} ({b})" for i, b in enumerate(blocks)]

    fig = go.Figure()
    max_parts = max((len(c) for c in contents), default=0)

    for part_idx in range(max_parts):
        xs = []
        texts = []
        hovers = []
        for b_idx in range(len(blocks)):
            if part_idx < len(contents[b_idx]):
                item = contents[b_idx][part_idx]
                xs.append(item["size"])
                texts.append(f"{item['request']} ({item['size']})")
                hovers.append(f"<b>{item['request']}</b><br>Block {b_idx + 1}<br>Size: {item['size']}")
            else:
                xs.append(0)
                texts.append("")
                hovers.append("")
        fig.add_trace(
            go.Bar(
                y=block_labels,
                x=xs,
                orientation="h",
                name=f"Segment {part_idx + 1}",
                showlegend=False,
                marker_color=PALETTE[part_idx % len(PALETTE)],
                marker_line_color=SURFACE,
                marker_line_width=1,
                text=texts,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(family=MONO, size=10, color="#FFFFFF"),
                hovertext=hovers,
                hoverinfo="text",
            )
        )

    fig.add_trace(
        go.Bar(
            y=block_labels,
            x=final_free,
            orientation="h",
            name="Free",
            showlegend=True,
            marker_color=FREE_FILL,
            marker_line_color=RULE,
            marker_line_width=1,
            text=[f"Free ({f})" if f > 0 else "" for f in final_free],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(family=MONO, size=10, color=MUTED),
            hovertemplate="<b>Free Space</b><br>%{y}<br>Remaining: %{x}<extra></extra>",
        )
    )

    style_fig(fig, title="Contiguous Block Partition Map", height=max(240, 48 * len(blocks) + 90))
    fig.update_layout(
        barmode="stack",
        xaxis_title="memory units",
        yaxis=dict(autorange="reversed", gridcolor=GRID, linecolor=IDLE, tickfont=dict(family=MONO, size=10, color=INK)),
    )
    return fig


def make_memory_comparison_chart(mem_results: dict[str, dict]) -> go.Figure:
    """Compare allocation count and utilization across First Fit, Best Fit, Worst Fit."""
    strats = list(mem_results.keys())
    labels = [s.replace("_", " ").title() for s in strats]
    utils = [round(mem_results[s]["utilization_pct"], 1) for s in strats]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=utils,
            marker_color=[PALETTE[0], PALETTE[1], PALETTE[3]],
            text=[f"{u:.1f}%" for u in utils],
            textposition="outside",
            textfont=dict(family=MONO, size=10, color=MUTED),
            showlegend=False,
        )
    )
    style_fig(fig, title="Memory Utilization by Strategy (%)", height=300)
    fig.update_layout(yaxis=dict(range=[0, 110], title="utilization (%)"))
    return fig


def make_disk_head_chart(disk_result: dict) -> go.Figure:
    """Track-vs-step trajectory showing disk head sweeps across cylinders."""
    path = disk_result["path"]
    steps = list(range(len(path)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=path,
            y=steps,
            mode="lines+markers+text",
            line=dict(color=PALETTE[0], width=2),
            marker=dict(color=PALETTE[0], size=7, line=dict(color=SURFACE, width=1.5)),
            text=[str(p) for p in path],
            textposition="top right",
            textfont=dict(family=MONO, size=10, color=INK),
            hovertemplate="Step %{y}<br>Track %{x}<extra></extra>",
            showlegend=False,
        )
    )
    style_fig(
        fig,
        title=f"Head Trajectory — {disk_result['algorithm'].upper()} (Total: {disk_result['total_movement']} tracks)",
        height=max(290, 34 * len(path) + 80),
    )
    fig.update_layout(
        xaxis=dict(
            title="cylinder / track",
            range=[-5, disk_result["disk_size"] + 5],
            gridcolor=GRID,
            linecolor=IDLE,
            tickfont=dict(family=MONO, size=10, color=FAINT),
        ),
        yaxis=dict(
            title="step",
            autorange="reversed",
            dtick=1,
            gridcolor=GRID,
            linecolor=IDLE,
            tickfont=dict(family=MONO, size=10, color=FAINT),
        ),
    )
    return fig


def make_disk_comparison_chart(disk_results: dict[str, dict]) -> go.Figure:
    """Bar chart comparing total head movement across FCFS, SSTF, SCAN."""
    algs = [a.upper() for a in disk_results.keys()]
    totals = [disk_results[a.lower()]["total_movement"] for a in algs]
    fig = go.Figure(
        go.Bar(
            x=algs,
            y=totals,
            marker_color=[PALETTE[0], PALETTE[3], PALETTE[2]],
            text=[f"{t} tracks" for t in totals],
            textposition="outside",
            textfont=dict(family=MONO, size=10, color=MUTED),
            showlegend=False,
        )
    )
    style_fig(fig, title="Total Head Movement Comparison (lower is better)", height=300)
    fig.update_layout(yaxis_title="tracks moved")
    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --paper:#FBFAF7; --surface:#FFFFFF; --ink:#1A1A1A; --muted:#5F5F5F;
  --faint:#8A8A8A; --rule:#DAD5C9; --rule-strong:#B9B2A3;
  --accent:#1F4E79; --accent-dark:#173B5C; --wash:#EAF0F6;
  --ok:#3F6B4F; --warn:#8C5A1B; --bad:#8C2F1B;
}

html, body, .stApp{background:var(--paper); color:var(--ink);}
.stApp{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"]{display:none;}
[data-testid="stDecoration"]{display:none;}
#MainMenu{visibility:hidden;}
[data-testid="stSidebar"]{display:none !important;}

.block-container{max-width:1120px; padding:2.2rem 2rem 4rem;}

/* top navigation bar */
.top-nav{
  display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px;
  padding:0.9rem 0 1.1rem; margin-bottom:0.6rem;
  border-bottom:1px solid var(--rule);
}
.top-brand{display:flex; align-items:center; gap:14px;}
.top-logo{
  font-family:'IBM Plex Mono',monospace; font-size:.82rem; letter-spacing:.12em;
  font-weight:600; color:var(--accent); border:1px solid var(--accent);
  padding:.22rem .5rem; border-radius:2px;
}
.top-title{
  font-family:'Source Serif 4',Georgia,serif; font-size:1.45rem; font-weight:600;
  color:var(--ink); letter-spacing:-.01em;
}
.top-sub{
  font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--faint);
}

h1,h2,h3,h4{font-family:'Source Serif 4',Georgia,serif !important; color:var(--ink) !important; font-weight:600 !important;}
h1{font-size:2.1rem; letter-spacing:-.01em; margin:.15rem 0 .35rem !important;}
h2{font-size:1.32rem; border-bottom:1px solid var(--rule); padding-bottom:.35rem; margin-top:2rem !important;}
h3{font-size:1.05rem; margin-top:1.3rem !important;}
p, li{font-size:.95rem; line-height:1.6; color:var(--muted);}

/* page header */
.page-head{border-bottom:1px solid var(--rule); padding-bottom:1rem; margin-bottom:1.5rem;}
.kicker{font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin-bottom:.35rem;}
.lede{color:var(--muted); font-size:.95rem; margin:.1rem 0 0; max-width:74ch;}

/* stat blocks — rules, not cards */
.stat{border-top:2px solid var(--ink); padding:.65rem 0 0;}
.stat-label{font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--faint);}
.stat-value{font-family:'Source Serif 4',Georgia,serif; font-size:1.7rem; font-weight:600;
  color:var(--ink); line-height:1.2; margin:.2rem 0 .1rem; font-variant-numeric:tabular-nums;}
.stat-value.ok{color:var(--ok);}
.stat-value.bad{color:var(--bad);}
.stat-note{font-family:'IBM Plex Mono',monospace; font-size:.66rem; color:var(--faint);}

/* panels */
.panel{background:var(--surface); border:1px solid var(--rule); border-radius:2px;
  padding:1.05rem 1.25rem; margin:.5rem 0 1rem;}
.panel p:last-child{margin-bottom:0;}

/* semantic callouts */
.callout{border:1px solid var(--rule); border-left:3px solid var(--accent); background:var(--surface);
  border-radius:2px; padding:.85rem 1.1rem; margin:.75rem 0 1.1rem;
  font-family:'IBM Plex Mono',monospace; font-size:.8rem; color:var(--ink);}
.callout.ok{border-left-color:var(--ok);}
.callout.bad{border-left-color:var(--bad);}
.callout.warn{border-left-color:var(--warn);}

/* controls */
.stButton > button, .stDownloadButton > button{
  border-radius:2px; font-family:'IBM Plex Mono',monospace;
  font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; min-height:38px;
  transition:background-color .12s ease, border-color .12s ease;
}
.stButton > button[kind="primary"]{background:var(--accent); border:1px solid var(--accent);}
.stButton > button[kind="primary"] p{color:#fff !important;}
.stButton > button[kind="primary"]:hover{background:var(--accent-dark); border-color:var(--accent-dark);}
.stButton > button[kind="secondary"], .stDownloadButton > button{
  background:transparent; border:1px solid var(--rule-strong); color:var(--ink);
}
.stButton > button[kind="secondary"] p, .stDownloadButton > button p{color:var(--ink) !important;}
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover{
  border-color:var(--accent); background:var(--wash);
}
.stButton > button[kind="secondary"]:hover p, .stDownloadButton > button:hover p{color:var(--accent) !important;}

[data-testid="stSelectbox"] label, [data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label, [data-testid="stTextArea"] label{
  font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--faint);
}
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea{
  border-radius:2px !important; border-color:var(--rule-strong) !important; background:var(--surface) !important;
  font-family:'IBM Plex Mono',monospace !important; font-size:.82rem !important;
}
[data-baseweb="select"] > div:focus-within, [data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus{
  border-color:var(--accent) !important; box-shadow:0 0 0 1px var(--accent) !important;
}

/* tabs — text with an underline, not pills */
.stTabs [data-baseweb="tab-list"]{gap:1.4rem; border-bottom:1px solid var(--rule);}
.stTabs [data-baseweb="tab"]{background:transparent; border-radius:0; padding:.45rem 0;
  font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--faint);}
.stTabs [aria-selected="true"]{color:var(--accent) !important; border-bottom:2px solid var(--accent);}

/* tables */
[data-testid="stDataFrame"]{border:1px solid var(--rule); border-radius:2px; overflow:hidden;}
[data-testid="stDataFrame"] *{font-family:'IBM Plex Mono',monospace !important; font-size:.78rem !important;}

hr{border:none; border-top:1px solid var(--rule); margin:1.6rem 0;}
.footer{font-family:'IBM Plex Mono',monospace; font-size:.66rem; color:var(--faint);
  border-top:1px solid var(--rule); padding-top:.8rem; margin-top:3rem; letter-spacing:.04em;}
.note{font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:var(--faint); line-height:1.6;}
</style>
"""

PAGES = [
    "Overview",
    "CPU Scheduling",
    "Memory Allocation",
    "Deadlock (Banker's)",
    "Disk Scheduling",
    "Dataset & Export",
]


def header(kicker: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="page-head"><div class="kicker">{kicker}</div>'
        f'<h1>{title}</h1><p class="lede">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def stat(label: str, value: str, note: str = "", tone: str = "") -> None:
    tone_cls = f" {tone}" if tone else ""
    st.markdown(
        f'<div class="stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value{tone_cls}">{value}</div>'
        f'<div class="stat-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def stat_row(items: list[tuple[str, str, str] | tuple[str, str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            if len(item) == 4:
                stat(item[0], item[1], item[2], item[3])
            else:
                stat(item[0], item[1], item[2])


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Enter at least one integer value.")
    return [int(p) for p in parts]


def compute_all_subsystems(ds: dict) -> dict:
    """Run CPU, Memory, Deadlock, and Disk simulations on the active dataset."""
    processes = [Process(**p) for p in ds["processes"]]
    quantum = int(ds.get("quantum", 2))
    cpu_results = {alg: cpu_schedule(processes, alg, quantum=quantum) for alg in ("fcfs", "sjf", "priority", "rr")}
    mem_results = {
        strat: mem_allocate(ds["memory"]["blocks"], ds["memory"]["requests"], strat)
        for strat in ("first_fit", "best_fit", "worst_fit")
    }
    dl_result = banker_safety(
        ds["deadlock"]["available"],
        ds["deadlock"]["allocation"],
        ds["deadlock"]["maximum"],
    )
    disk_results = {
        alg: disk_schedule(
            ds["disk"]["requests"],
            int(ds["disk"]["head"]),
            alg,
            int(ds["disk"].get("disk_size", 200)),
            ds["disk"].get("direction", "right"),
        )
        for alg in ("fcfs", "sstf", "scan")
    }
    return {
        "scheduling": cpu_results,
        "memory": mem_results,
        "deadlock": dl_result,
        "disk": disk_results,
    }


def render_app() -> None:
    """Render the Streamlit OS Resource Studio interface."""
    st.set_page_config(
        page_title="OS Resource Studio",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    if "dataset" not in st.session_state:
        st.session_state.dataset = get_preset("Lab Standard (Default)")
    if "preset_name" not in st.session_state:
        st.session_state.preset_name = "Lab Standard (Default)"
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    if "cpu_alg" not in st.session_state:
        st.session_state.cpu_alg = "fcfs"
    if "mem_strat" not in st.session_state:
        st.session_state.mem_strat = "first_fit"
    if "disk_alg" not in st.session_state:
        st.session_state.disk_alg = "sstf"

    # Top Navigation Bar
    st.markdown(
        '<div class="top-nav">'
        '<div class="top-brand">'
        '<span class="top-logo">OS</span>'
        '<span class="top-title">OS Resource Studio</span>'
        '</div>'
        '<div class="top-sub" style="text-align:right; color:var(--muted);">'
        'Name: Divesh Kinha &nbsp;·&nbsp; Roll No: 2401010077 &nbsp;·&nbsp; B.Tech CSE Core'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    nav_cols = st.columns(len(PAGES))
    for col, pname in zip(nav_cols, PAGES):
        with col:
            active = st.session_state.page == pname
            label = f"● {pname}" if active else pname
            if st.button(
                label,
                key=f"nav_{pname}",
                type="primary" if active else "secondary",
                width="stretch",
            ):
                st.session_state.page = pname
                st.rerun()

    page = st.session_state.page
    ds = st.session_state.dataset

    # ========================================================================
    # PAGE 1: OVERVIEW
    # ========================================================================
    if page == "Overview":
        header(
            "Integrated Summary",
            "Operating System Resource Management",
            "Unified comparison across CPU scheduling, contiguous memory allocation, Banker's deadlock avoidance, and disk head scheduling on a shared workload.",
        )

        ctrl_l, ctrl_r = st.columns([3, 1])
        with ctrl_l:
            preset_choice = st.selectbox(
                "Workload Preset",
                list(PRESETS.keys()),
                index=list(PRESETS.keys()).index(st.session_state.preset_name)
                if st.session_state.preset_name in PRESETS
                else 0,
            )
            if preset_choice != st.session_state.preset_name:
                st.session_state.preset_name = preset_choice
                st.session_state.dataset = get_preset(preset_choice)
                st.rerun()
        with ctrl_r:
            st.write("")
            st.write("")
            preview_alg = st.selectbox("Hero Gantt Policy", ["fcfs", "sjf", "priority", "rr"], format_func=str.upper)

        try:
            all_res = compute_all_subsystems(ds)
            cpu_res = all_res["scheduling"]
            mem_res = all_res["memory"]
            dl_res = all_res["deadlock"]
            disk_res = all_res["disk"]

            best_cpu = min(cpu_res.items(), key=lambda kv: kv[1]["avg_waiting"])
            best_mem = max(mem_res.items(), key=lambda kv: (kv[1]["allocated_count"], kv[1]["utilization_pct"]))
            best_disk = min(disk_res.items(), key=lambda kv: kv[1]["total_movement"])
            dl_state = "SAFE" if dl_res["safe"] else "UNSAFE"
            dl_note = f"seq {dl_res['sequence']}" if dl_res["safe"] else f"unfinished {dl_res['unfinished']}"

            stat_row(
                [
                    ("Lowest CPU Wait", f"{best_cpu[1]['avg_waiting']:.2f}", f"{best_cpu[0].upper()} · {best_cpu[1]['cpu_utilization']:.1f}% util"),
                    ("Best Memory Fit", f"{best_mem[1]['allocated_count']}/{len(best_mem[1]['allocations'])}", f"{best_mem[0].replace('_', ' ')} · {best_mem[1]['utilization_pct']:.1f}%"),
                    ("Banker's State", dl_state, dl_note, "ok" if dl_res["safe"] else "bad"),
                    ("Lowest Disk Seek", f"{best_disk[1]['total_movement']}", f"{best_disk[0].upper()} · {best_disk[1]['average_movement']:.1f} avg"),
                ]
            )

            st.markdown(f"### Execution Trace — {preview_alg.upper()}")
            st.plotly_chart(
                make_gantt(cpu_res[preview_alg]["segments"]),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": True},
            )

            c_left, c_right = st.columns(2)
            with c_left:
                st.plotly_chart(
                    make_memory_comparison_chart(mem_res),
                    width="stretch",
                    config={"displaylogo": False},
                )
            with c_right:
                st.plotly_chart(
                    make_disk_comparison_chart(disk_res),
                    width="stretch",
                    config={"displaylogo": False},
                )

            st.markdown("### Subsystem Export")
            ex1, ex2 = st.columns([3, 1])
            with ex1:
                st.markdown(
                    '<div class="panel"><p>Export the complete computed analysis across all four subsystems as a structured JSON report, or switch to any module tab above to edit parameters interactively.</p></div>',
                    unsafe_allow_html=True,
                )
            with ex2:
                st.download_button(
                    "Download analysis.json",
                    data=json.dumps(all_res, indent=2),
                    file_name="analysis.json",
                    mime="application/json",
                    width="stretch",
                )
        except Exception as exc:
            st.error(f"Simulation error: {exc}")

    # ========================================================================
    # PAGE 2: CPU SCHEDULING
    # ========================================================================
    elif page == "CPU Scheduling":
        header(
            "Subsystem 01 · CPU",
            "CPU Scheduling & Gantt Trace",
            "Edit the process table directly to compare First-Come First-Served, Shortest Job First, Priority (lower value = higher priority), and preemptive Round Robin.",
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            alg = st.selectbox(
                "Active Policy",
                ["fcfs", "sjf", "priority", "rr"],
                index=["fcfs", "sjf", "priority", "rr"].index(st.session_state.cpu_alg),
                format_func=lambda a: {
                    "fcfs": "FCFS — First-Come, First-Served",
                    "sjf": "SJF — Shortest Job First (Non-preemptive)",
                    "priority": "Priority — Non-preemptive (Lower = Higher)",
                    "rr": "Round Robin — Preemptive Time-Sliced",
                }[a],
            )
            st.session_state.cpu_alg = alg
        with c2:
            quantum = st.number_input("Round Robin Quantum", min_value=1, max_value=50, value=int(ds.get("quantum", 2)), step=1)
            ds["quantum"] = int(quantum)

        proc_df = pd.DataFrame(ds["processes"])
        edited_df = st.data_editor(
            proc_df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "pid": st.column_config.TextColumn("Process ID", required=True),
                "arrival": st.column_config.NumberColumn("Arrival Time", min_value=0, step=1, required=True),
                "burst": st.column_config.NumberColumn("Burst Time", min_value=1, step=1, required=True),
                "priority": st.column_config.NumberColumn("Priority", min_value=0, step=1, required=True),
            },
            key="cpu_editor",
        )

        try:
            clean_procs = []
            for _, row in edited_df.dropna(how="all").iterrows():
                pid = str(row.get("pid", "") or "").strip()
                if not pid:
                    continue
                clean_procs.append(
                    {
                        "pid": pid,
                        "arrival": int(row["arrival"]) if pd.notna(row["arrival"]) else 0,
                        "burst": int(row["burst"]) if pd.notna(row["burst"]) else 1,
                        "priority": int(row["priority"]) if pd.notna(row["priority"]) else 0,
                    }
                )
            ds["processes"] = clean_procs
            processes = [Process(**p) for p in clean_procs]
            cpu_results = {a: cpu_schedule(processes, a, quantum=int(quantum)) for a in ("fcfs", "sjf", "priority", "rr")}
            active_res = cpu_results[alg]

            stat_row(
                [
                    ("Avg Waiting", f"{active_res['avg_waiting']:.2f}", "time units"),
                    ("Avg Turnaround", f"{active_res['avg_turnaround']:.2f}", "time units"),
                    ("Avg Response", f"{active_res['avg_response']:.2f}", "time units"),
                    ("CPU Utilization", f"{active_res['cpu_utilization']:.1f}%", f"makespan {active_res['makespan']}"),
                    ("Throughput", f"{active_res['throughput']:.3f}", "proc / unit"),
                    ("Context Switches", str(active_res["context_switches"]), "burst transitions"),
                ]
            )

            st.markdown(f"### Gantt Chart — {alg.upper()}")
            st.plotly_chart(
                make_gantt(active_res["segments"]),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": True},
            )

            st.markdown("### All-Policy Comparison")
            summary_rows = [
                {
                    "Algorithm": a.upper(),
                    "Avg Waiting": round(r["avg_waiting"], 2),
                    "Avg Turnaround": round(r["avg_turnaround"], 2),
                    "Avg Response": round(r["avg_response"], 2),
                    "CPU Util (%)": round(r["cpu_utilization"], 1),
                    "Makespan": r["makespan"],
                    "Context Switches": r["context_switches"],
                }
                for a, r in cpu_results.items()
            ]
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
            st.plotly_chart(make_cpu_comparison_chart(cpu_results), width="stretch", config={"displaylogo": False})
        except Exception as exc:
            st.error(f"CPU Scheduling error: {exc}")

    # ========================================================================
    # PAGE 3: MEMORY ALLOCATION
    # ========================================================================
    elif page == "Memory Allocation":
        header(
            "Subsystem 02 · Memory",
            "Contiguous Memory Allocation",
            "Simulate First Fit, Best Fit, and Worst Fit partitioning across fixed free blocks, inspect fragmentation, and visualize block occupancy.",
        )

        c1, c2, c3 = st.columns([2, 2, 1.5])
        with c1:
            blocks_text = st.text_input(
                "Free Memory Blocks (comma-separated)",
                value=", ".join(map(str, ds["memory"]["blocks"])),
            )
        with c2:
            reqs_text = st.text_input(
                "Incoming Requests (comma-separated)",
                value=", ".join(map(str, ds["memory"]["requests"])),
            )
        with c3:
            strat = st.selectbox(
                "Allocation Strategy",
                ["first_fit", "best_fit", "worst_fit"],
                index=["first_fit", "best_fit", "worst_fit"].index(st.session_state.mem_strat),
                format_func=lambda s: s.replace("_", " ").title(),
            )
            st.session_state.mem_strat = strat

        try:
            blocks = parse_int_list(blocks_text)
            mem_reqs = parse_int_list(reqs_text)
            ds["memory"]["blocks"] = blocks
            ds["memory"]["requests"] = mem_reqs

            mem_results = {s: mem_allocate(blocks, mem_reqs, s) for s in ("first_fit", "best_fit", "worst_fit")}
            active_mem = mem_results[strat]

            stat_row(
                [
                    ("Requests Allocated", f"{active_mem['allocated_count']}/{len(active_mem['allocations'])}", "fulfilled requests"),
                    ("Memory Utilization", f"{active_mem['utilization_pct']:.1f}%", f"{active_mem['total_allocated']} / {active_mem['total_capacity']} units"),
                    ("Remaining Free", f"{active_mem['total_free']}", "unallocated capacity"),
                    ("Unallocated", str(active_mem["unallocated_count"]), "rejected requests", "bad" if active_mem["unallocated_count"] > 0 else "ok"),
                ]
            )

            st.plotly_chart(make_memory_map(active_mem), width="stretch", config={"displaylogo": False})

            t_trace, t_cmp = st.tabs(["Allocation Trace", "Strategy Comparison"])
            with t_trace:
                trace_df = pd.DataFrame(active_mem["allocations"])
                trace_df["block"] = trace_df["block"].apply(lambda b: f"Block {int(b)}" if pd.notna(b) else "—")
                trace_df["remaining_in_block"] = trace_df["remaining_in_block"].apply(lambda v: str(int(v)) if pd.notna(v) else "—")
                st.dataframe(trace_df, width="stretch", hide_index=True)
            with t_cmp:
                cmp_rows = [
                    {
                        "Strategy": s.replace("_", " ").title(),
                        "Allocated": f"{r['allocated_count']}/{len(r['allocations'])}",
                        "Utilization (%)": round(r["utilization_pct"], 1),
                        "Remaining Free": r["total_free"],
                        "Final Free Vector": str(r["final_free"]),
                    }
                    for s, r in mem_results.items()
                ]
                st.dataframe(pd.DataFrame(cmp_rows), width="stretch", hide_index=True)
                st.plotly_chart(make_memory_comparison_chart(mem_results), width="stretch", config={"displaylogo": False})
        except Exception as exc:
            st.error(f"Memory Allocation error: {exc}")

    # ========================================================================
    # PAGE 4: DEADLOCK (BANKER'S)
    # ========================================================================
    elif page == "Deadlock (Banker's)":
        header(
            "Subsystem 03 · Deadlock Avoidance",
            "Banker's Safety Algorithm",
            "Inspect the system's safe completion sequence, examine how the Work vector evolves step by step, and test hypothetical resource requests.",
        )

        dl = ds["deadlock"]
        n_procs = len(dl["allocation"])
        n_res = len(dl["available"])
        res_cols = [f"R{j}" for j in range(n_res)]
        proc_idx = [f"P{i}" for i in range(n_procs)]

        avail_col, alloc_col, max_col = st.columns([1, 1.4, 1.4])
        with avail_col:
            avail_text = st.text_input("Available Vector", value=", ".join(map(str, dl["available"])))
        with alloc_col:
            st.markdown('<div class="note">Allocation Matrix</div>', unsafe_allow_html=True)
            alloc_df = pd.DataFrame(dl["allocation"], columns=res_cols, index=proc_idx)
            edited_alloc = st.data_editor(alloc_df, width="stretch", key="alloc_editor")
        with max_col:
            st.markdown('<div class="note">Maximum Matrix</div>', unsafe_allow_html=True)
            max_df = pd.DataFrame(dl["maximum"], columns=res_cols, index=proc_idx)
            edited_max = st.data_editor(max_df, width="stretch", key="max_editor")

        try:
            available = parse_int_list(avail_text)
            allocation = edited_alloc.astype(int).values.tolist()
            maximum = edited_max.astype(int).values.tolist()
            dl["available"] = available
            dl["allocation"] = allocation
            dl["maximum"] = maximum

            dl_res = banker_safety(available, allocation, maximum)

            if dl_res["safe"]:
                seq_str = " → ".join(f"P{i}" for i in dl_res["sequence"])
                st.markdown(
                    f'<div class="callout ok"><strong>SYSTEM SAFE</strong> — Completion sequence: {seq_str}</div>',
                    unsafe_allow_html=True,
                )
            else:
                unfin = ", ".join(f"P{i}" for i in dl_res["unfinished"])
                st.markdown(
                    f'<div class="callout bad"><strong>SYSTEM UNSAFE</strong> — No safe completion order; unfinished processes: {unfin}</div>',
                    unsafe_allow_html=True,
                )

            t_need, t_trace, t_req = st.tabs(["Need Matrix", "Safety Execution Trace", "Test Resource Request"])
            with t_need:
                need_df = pd.DataFrame(dl_res["need"], columns=res_cols, index=proc_idx)
                st.dataframe(need_df, width="stretch")
            with t_trace:
                if dl_res["trace"]:
                    trace_rows = [
                        {
                            "Step": t["step"],
                            "Process": t["process"],
                            "Need": str(t["need"]),
                            "Work Before": str(t["work_before"]),
                            "Allocation": str(t["allocation"]),
                            "Work After": str(t["work_after"]),
                        }
                        for t in dl_res["trace"]
                    ]
                    st.dataframe(pd.DataFrame(trace_rows), width="stretch", hide_index=True)
                else:
                    st.markdown('<div class="note">No process could satisfy its remaining Need with the initial Available vector.</div>', unsafe_allow_html=True)
            with t_req:
                rc1, rc2, rc3 = st.columns([1, 2, 1])
                with rc1:
                    req_proc = st.number_input("Process Index (0-based)", min_value=0, max_value=max(0, len(allocation) - 1), value=0, step=1)
                with rc2:
                    req_vec_text = st.text_input("Request Vector (comma-separated)", value=", ".join(["0"] * len(available)))
                with rc3:
                    st.write("")
                    st.write("")
                    check_btn = st.button("Evaluate Request", type="primary", width="stretch")
                if check_btn:
                    req_vec = parse_int_list(req_vec_text)
                    req_res = request_resources(available, allocation, maximum, int(req_proc), req_vec)
                    if req_res["granted"]:
                        seq = " → ".join(f"P{i}" for i in req_res["safety"]["sequence"])
                        st.markdown(
                            f'<div class="callout ok"><strong>REQUEST GRANTED</strong> — {req_res["reason"]}. New safe sequence: {seq}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="callout warn"><strong>REQUEST DENIED</strong> — {req_res["reason"]}</div>',
                            unsafe_allow_html=True,
                        )
        except Exception as exc:
            st.error(f"Deadlock check error: {exc}")

    # ========================================================================
    # PAGE 5: DISK SCHEDULING
    # ========================================================================
    elif page == "Disk Scheduling":
        header(
            "Subsystem 04 · Storage",
            "Disk Head Scheduling",
            "Compare FCFS, Shortest Seek Time First (SSTF), and directional SCAN sweep across cylinder requests.",
        )

        dsk = ds["disk"]
        c1, c2, c3, c4, c5 = st.columns([2.4, 1, 1, 1, 1.2])
        with c1:
            disk_reqs_text = st.text_input("Track Requests (comma-separated)", value=", ".join(map(str, dsk["requests"])))
        with c2:
            head_pos = st.number_input("Initial Head", min_value=0, value=int(dsk["head"]), step=1)
        with c3:
            disk_size = st.number_input("Disk Size", min_value=1, value=int(dsk.get("disk_size", 200)), step=1)
        with c4:
            direction = st.selectbox("SCAN Direction", ["right", "left"], index=0 if dsk.get("direction", "right") == "right" else 1)
        with c5:
            disk_alg = st.selectbox(
                "Inspect Policy",
                ["fcfs", "sstf", "scan"],
                index=["fcfs", "sstf", "scan"].index(st.session_state.disk_alg),
                format_func=str.upper,
            )
            st.session_state.disk_alg = disk_alg

        try:
            disk_reqs = parse_int_list(disk_reqs_text)
            dsk["requests"] = disk_reqs
            dsk["head"] = int(head_pos)
            dsk["disk_size"] = int(disk_size)
            dsk["direction"] = direction

            disk_results = {a: disk_schedule(disk_reqs, int(head_pos), a, int(disk_size), direction) for a in ("fcfs", "sstf", "scan")}
            active_disk = disk_results[disk_alg]
            best_disk = min(disk_results.items(), key=lambda kv: kv[1]["total_movement"])

            stat_row(
                [
                    ("Active Policy", disk_alg.upper(), f"direction: {direction}"),
                    ("Total Head Movement", f"{active_disk['total_movement']}", "tracks traversed"),
                    ("Average Seek", f"{active_disk['average_movement']:.2f}", "tracks / request"),
                    ("Best on Workload", best_disk[0].upper(), f"{best_disk[1]['total_movement']} tracks"),
                ]
            )

            st.plotly_chart(make_disk_head_chart(active_disk), width="stretch", config={"displaylogo": False})

            t_steps, t_cmp = st.tabs(["Step-by-Step Seek Trace", "All-Policy Comparison"])
            with t_steps:
                st.dataframe(pd.DataFrame(active_disk["step_distances"]), width="stretch", hide_index=True)
            with t_cmp:
                cmp_rows = [
                    {
                        "Algorithm": a.upper(),
                        "Total Movement": r["total_movement"],
                        "Average Seek": round(r["average_movement"], 2),
                        "Service Order": str(r["order"]),
                    }
                    for a, r in disk_results.items()
                ]
                st.dataframe(pd.DataFrame(cmp_rows), width="stretch", hide_index=True)
                st.plotly_chart(make_disk_comparison_chart(disk_results), width="stretch", config={"displaylogo": False})
        except Exception as exc:
            st.error(f"Disk Scheduling error: {exc}")

    # ========================================================================
    # PAGE 6: DATASET & EXPORT
    # ========================================================================
    else:
        header(
            "Configuration & Artifacts",
            "Dataset & Export",
            "Load a curated workload, upload or edit the unified JSON contract, and export CSV summaries or the full analysis bundle.",
        )

        p1, p2 = st.columns([2, 2])
        with p1:
            selected_preset = st.selectbox("Load Curated Preset", list(PRESETS.keys()))
            if st.button("Apply Preset", type="primary"):
                st.session_state.preset_name = selected_preset
                st.session_state.dataset = get_preset(selected_preset)
                st.rerun()
        with p2:
            uploaded = st.file_uploader("Upload Custom JSON Dataset", type=["json"])
            if uploaded is not None:
                try:
                    loaded = json.loads(uploaded.read().decode("utf-8"))
                    compute_all_subsystems(loaded)
                    st.session_state.dataset = loaded
                    st.session_state.preset_name = uploaded.name
                    st.success(f"Loaded {uploaded.name}")
                except Exception as exc:
                    st.error(f"Invalid dataset JSON: {exc}")

        st.markdown("### Unified JSON Contract")
        raw_json = st.text_area("Active Dataset JSON", value=json.dumps(ds, indent=2), height=280)
        if st.button("Apply JSON Changes", type="secondary"):
            try:
                parsed = json.loads(raw_json)
                compute_all_subsystems(parsed)
                st.session_state.dataset = parsed
                st.session_state.preset_name = "Custom JSON"
                st.success("Dataset updated.")
            except Exception as exc:
                st.error(f"Validation failed: {exc}")

        st.markdown("### Export Reports")
        try:
            all_res = compute_all_subsystems(st.session_state.dataset)
            sched_df = pd.DataFrame(
                [
                    {
                        "algorithm": a,
                        "avg_waiting": round(r["avg_waiting"], 4),
                        "avg_turnaround": round(r["avg_turnaround"], 4),
                        "avg_response": round(r["avg_response"], 4),
                        "makespan": r["makespan"],
                        "cpu_utilization": round(r["cpu_utilization"], 2),
                    }
                    for a, r in all_res["scheduling"].items()
                ]
            )
            mem_df = pd.DataFrame(
                [
                    {
                        "strategy": s,
                        "allocated_count": r["allocated_count"],
                        "total_requests": len(r["allocations"]),
                        "utilization_pct": round(r["utilization_pct"], 2),
                        "total_free": r["total_free"],
                    }
                    for s, r in all_res["memory"].items()
                ]
            )
            disk_df = pd.DataFrame(
                [
                    {
                        "algorithm": a,
                        "total_movement": r["total_movement"],
                        "average_movement": round(r["average_movement"], 4),
                        "order": str(r["order"]),
                    }
                    for a, r in all_res["disk"].items()
                ]
            )

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.download_button(
                    "analysis.json",
                    data=json.dumps(all_res, indent=2),
                    file_name="analysis.json",
                    mime="application/json",
                    width="stretch",
                )
            with d2:
                st.download_button(
                    "scheduling.csv",
                    data=sched_df.to_csv(index=False),
                    file_name="scheduling.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with d3:
                st.download_button(
                    "memory.csv",
                    data=mem_df.to_csv(index=False),
                    file_name="memory.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with d4:
                st.download_button(
                    "disk.csv",
                    data=disk_df.to_csv(index=False),
                    file_name="disk.csv",
                    mime="text/csv",
                    width="stretch",
                )
        except Exception as exc:
            st.error(f"Export unavailable: {exc}")

    st.markdown(
        '<div class="footer">OS Resource Studio · CPU Scheduling, Contiguous Memory, Banker\'s Deadlock Avoidance & Disk Head Scheduling</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# UNIT TESTS (INTEGRATED)
# ============================================================================
class TestScheduler(unittest.TestCase):
    def setUp(self) -> None:
        ds = get_preset("Lab Standard (Default)")
        self.procs = [Process(**p) for p in ds["processes"]]

    def test_sample_metrics(self) -> None:
        fcfs = cpu_schedule(self.procs, "fcfs")
        sjf = cpu_schedule(self.procs, "sjf")
        prio = cpu_schedule(self.procs, "priority")
        rr = cpu_schedule(self.procs, "rr", quantum=2)

        self.assertAlmostEqual(fcfs["avg_waiting"], 4.75)
        self.assertAlmostEqual(sjf["avg_waiting"], 4.00)
        self.assertAlmostEqual(prio["avg_waiting"], 5.50)
        self.assertAlmostEqual(rr["avg_waiting"], 5.00)
        self.assertAlmostEqual(rr["avg_response"], 1.50)

    def test_idle_gap_segments(self) -> None:
        procs = [Process("P1", 2, 3, 1), Process("P2", 8, 2, 1)]
        res = cpu_schedule(procs, "fcfs")
        pids = [s["pid"] for s in res["segments"]]
        self.assertEqual(pids, ["IDLE", "P1", "IDLE", "P2"])
        self.assertLess(res["cpu_utilization"], 100.0)


class TestMemory(unittest.TestCase):
    def test_strategies(self) -> None:
        blocks = [100, 500, 200, 300, 600]
        reqs = [212, 417, 112, 426, 95]
        ff = mem_allocate(blocks, reqs, "first_fit")
        bf = mem_allocate(blocks, reqs, "best_fit")
        wf = mem_allocate(blocks, reqs, "worst_fit")

        self.assertEqual(ff["allocated_count"], 4)
        self.assertEqual(bf["allocated_count"], 5)
        self.assertEqual(wf["allocated_count"], 4)
        self.assertEqual(len(bf["block_contents"]), len(blocks))


class TestDeadlock(unittest.TestCase):
    def test_safe_and_request(self) -> None:
        ds = get_preset("Lab Standard (Default)")["deadlock"]
        res = banker_safety(ds["available"], ds["allocation"], ds["maximum"])
        self.assertTrue(res["safe"])
        self.assertEqual(res["sequence"], [1, 3, 4, 0, 2])
        self.assertEqual(len(res["trace"]), 5)

        grant = request_resources(ds["available"], ds["allocation"], ds["maximum"], 1, [1, 0, 2])
        self.assertTrue(grant["granted"])

        deny = request_resources(ds["available"], ds["allocation"], ds["maximum"], 0, [3, 3, 0])
        self.assertFalse(deny["granted"])

    def test_unsafe_preset(self) -> None:
        ds = get_preset("Unsafe Deadlock & Fragmentation")["deadlock"]
        res = banker_safety(ds["available"], ds["allocation"], ds["maximum"])
        self.assertFalse(res["safe"])
        self.assertTrue(len(res["unfinished"]) > 0)


class TestDisk(unittest.TestCase):
    def test_sample_disk(self) -> None:
        ds = get_preset("Lab Standard (Default)")["disk"]
        fcfs = disk_schedule(ds["requests"], ds["head"], "fcfs", ds["disk_size"], ds["direction"])
        sstf = disk_schedule(ds["requests"], ds["head"], "sstf", ds["disk_size"], ds["direction"])
        scan = disk_schedule(ds["requests"], ds["head"], "scan", ds["disk_size"], ds["direction"])

        self.assertEqual(fcfs["total_movement"], 640)
        self.assertEqual(sstf["total_movement"], 236)
        self.assertEqual(scan["total_movement"], 331)


class TestChartsAndPresets(unittest.TestCase):
    def test_all_presets_and_charts(self) -> None:
        for name in PRESETS:
            ds = get_preset(name)
            procs = [Process(**p) for p in ds["processes"]]
            cpu_res = {a: cpu_schedule(procs, a, ds["quantum"]) for a in ("fcfs", "sjf", "priority", "rr")}
            mem_res = {s: mem_allocate(ds["memory"]["blocks"], ds["memory"]["requests"], s) for s in ("first_fit", "best_fit", "worst_fit")}
            disk_res = {a: disk_schedule(ds["disk"]["requests"], ds["disk"]["head"], a, ds["disk"]["disk_size"], ds["disk"]["direction"]) for a in ("fcfs", "sstf", "scan")}

            self.assertIsNotNone(make_gantt(cpu_res["fcfs"]["segments"]))
            self.assertIsNotNone(make_cpu_comparison_chart(cpu_res))
            self.assertIsNotNone(make_memory_map(mem_res["best_fit"]))
            self.assertIsNotNone(make_memory_comparison_chart(mem_res))
            self.assertIsNotNone(make_disk_head_chart(disk_res["sstf"]))
            self.assertIsNotNone(make_disk_comparison_chart(disk_res))


# ============================================================================
# ENTRY POINT
# ============================================================================
if _RUN_TESTS_ON_MAIN:
    unittest.main(verbosity=2)
elif get_script_run_ctx() is not None:
    render_app()
