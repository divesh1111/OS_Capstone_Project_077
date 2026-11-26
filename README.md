# OS Resource Studio

**Author:** Divesh Kinha · **Roll No:** 2401010077 · **Program:** B.Tech CSE Core

An interactive, academic-styled Operating Systems Resource Management Studio built with **Python**, **Streamlit**, and **Plotly**. It unifies four core OS subsystems—**CPU Scheduling**, **Contiguous Memory Allocation**, **Deadlock Avoidance (Banker's Algorithm)**, and **Disk Head Scheduling**—into a single, reproducible workbench driven by a shared JSON workload contract.

---

## Table of Contents

1. [Features & Design System](#features--design-system)
2. [Project Structure](#project-structure)
3. [Setup & Running](#setup--running)
4. [Studio Pages & Modules](#studio-pages--modules)
   - [1. Overview](#1-overview)
   - [2. CPU Scheduling (`core/scheduler.py`)](#2-cpu-scheduling-coreschedulerpy)
   - [3. Contiguous Memory Allocation (`core/memory.py`)](#3-contiguous-memory-allocation-corememorypy)
   - [4. Deadlock Avoidance — Banker's Algorithm (`core/deadlock.py`)](#4-deadlock-avoidance--bankers-algorithm-coredeadlockpy)
   - [5. Disk Head Scheduling (`core/disk.py`)](#5-disk-head-scheduling-corediskpy)
   - [6. Dataset & Export (`core/presets.py`)](#6-dataset--export-corepresetspy)
5. [Unified JSON Input Contract](#unified-json-input-contract)
6. [Visualization Engine (`utils/charts.py`)](#visualization-engine-utilschartspy)
7. [Automated Testing (`tests/test_core.py`)](#automated-testing-teststest_corepy)

---

## Features & Design System

The interface is styled after a printed academic lab report:
- **Color Palette:** Warm paper-white canvas (`#FBFAF7`), pure white panels (`#FFFFFF`), ink-black headings (`#1A1A1A`), hairline rules (`#DAD5C9`), and an academic navy accent (`#1F4E79`). Semantic callouts use moss green (`#3F6B4F`), ochre (`#8C5A1B`), and brick red (`#8C2F1B`).
- **Typography:** `Source Serif 4` for headings and metric values, `Inter` for body prose, and `IBM Plex Mono` (`tabular-nums`) for labels, tables, and chart axes.
- **Print-Safe Charts:** All Plotly figures use white backgrounds, hairline grids (`#E7E2D8`), direct annotations, and a muted 8-color categorical palette with `#B9B2A3` reserved for idle/free segments.

---

## Project Structure

```text
OS_Resource_Studio/
├── .streamlit/
│   └── config.toml          # Light paper-white theme & typography config
├── core/
│   ├── __init__.py
│   ├── scheduler.py         # CPU scheduling engines (FCFS, SJF, Priority, RR)
│   ├── memory.py            # Contiguous memory fit strategies (First, Best, Worst)
│   ├── deadlock.py          # Banker's safety check & resource request validator
│   ├── disk.py              # Disk head scheduling (FCFS, SSTF, SCAN)
│   └── presets.py           # Curated multi-subsystem workload presets
├── utils/
│   ├── __init__.py
│   └── charts.py            # Academic-styled Plotly figure builders
├── tests/
│   └── test_core.py         # Unit tests for all engines, presets, and charts
├── app.py                   # Main Streamlit application & direct-launch entry point
├── run.sh                   # Shell launcher script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Setup & Running

### Prerequisites
- Python 3.9+
- Packages listed in `requirements.txt`:
  - `streamlit>=1.50`
  - `plotly>=5.22`
  - `pandas>=2.2`
  - `numpy>=1.26`

Install dependencies if needed:

```bash
cd OS_Resource_Studio
pip install -r requirements.txt
```

### Launch the Application

You can run the studio directly with Python:

```bash
cd OS_Resource_Studio
python3 app.py
```

When invoked directly (`python3 app.py`), `app.py` detects that no active Streamlit `ScriptRunContext` exists and automatically boots the Streamlit server.

Alternatively, you can use the Streamlit CLI or helper script:

```bash
streamlit run app.py
# or
./run.sh
```

Once started, open **http://localhost:8501** in your browser.

---

## Studio Pages & Modules

### 1. Overview
Provides an integrated executive summary across all four subsystems on the active dataset:
- **4 Top-Ruled Stat Blocks:** Lowest CPU Avg Wait, Best Memory Fit (allocated requests & utilization %), Banker's Safety State (`SAFE` / `UNSAFE` with sequence or unfinished processes), and Lowest Disk Seek movement.
- **Hero Execution Trace:** Full-width interactive CPU Gantt chart with a policy selector (`FCFS`, `SJF`, `Priority`, `RR`).
- **Subsystem Snapshots:** Side-by-side Memory Utilization and Disk Head Movement comparison charts.
- **Preset Switcher & Export:** Switch between curated workloads or download the complete computed `analysis.json`.

---

### 2. CPU Scheduling (`core/scheduler.py`)
Simulates and compares four classical CPU scheduling policies on a live-editable process table (`pid`, `arrival`, `burst`, `priority`):

| Algorithm | Type | Selection Rule |
|---|---|---|
| **FCFS** | Non-preemptive | Earliest arrival time (ties broken by input order) |
| **SJF** | Non-preemptive | Shortest burst time among ready processes |
| **Priority** | Non-preemptive | Lowest integer priority value among ready processes (`0` / `1` = highest) |
| **Round Robin (RR)** | Preemptive | FIFO ready queue with configurable time `quantum` |

**Computed Metrics & Outputs:**
- **Gantt Chart:** Single-row timeline with `IDLE` gaps inserted automatically whenever the CPU waits for the next arrival.
- **Summary Stat Blocks:**
  - `Avg Waiting` = $\frac{1}{n}\sum (\text{Turnaround} - \text{Burst})$
  - `Avg Turnaround` = $\frac{1}{n}\sum (\text{Completion} - \text{Arrival})$
  - `Avg Response` = $\frac{1}{n}\sum (\text{First Start} - \text{Arrival})$
  - `CPU Utilization` = $\frac{\sum \text{Burst}}{\text{Makespan}} \times 100\%$
  - `Throughput` = $\frac{n}{\text{Makespan}}$ (processes per time unit)
  - `Context Switches` = number of transitions between non-idle execution segments
- **All-Policy Comparison:** Summary table and grouped bar chart comparing all four algorithms side by side.

---

### 3. Contiguous Memory Allocation (`core/memory.py`)
Models variable-size memory requests arriving sequentially against fixed contiguous free blocks:

| Strategy | Behavior |
|---|---|
| **First Fit** | Allocates the first block whose remaining free space $\ge$ request size |
| **Best Fit** | Allocates the smallest block whose remaining free space $\ge$ request size |
| **Worst Fit** | Allocates the largest block whose remaining free space $\ge$ request size |

- Successful allocations deduct the request size from the chosen block's remaining free space; failed requests leave free blocks unchanged and are flagged as unallocated.
- **Visualizations:**
  - **Contiguous Block Partition Map:** Horizontal stacked bar chart showing each request (`R1`, `R2`, …) placed inside its assigned block alongside remaining free space.
  - **Allocation Trace & Strategy Comparison:** Tabular trace per request and bar chart comparing memory utilization (%) across First Fit, Best Fit, and Worst Fit.

---

### 4. Deadlock Avoidance — Banker's Algorithm (`core/deadlock.py`)
Evaluates system safety and tests hypothetical resource requests using editable matrices:

- **Inputs:**
  - `Available` vector ($1 \times m$)
  - `Allocation` matrix ($n \times m$)
  - `Maximum` matrix ($n \times m$)
- **Safety Check (`safety()`):**
  1. Computes $\text{Need}[i][j] = \text{Maximum}[i][j] - \text{Allocation}[i][j]$.
  2. Initializes `Work = Available` and `Finish = [False] * n`.
  3. Repeatedly finds an unfinished process $P_i$ with $\text{Need}_i \le \text{Work}$, releases its `Allocation` back into `Work`, and records each step in `trace`.
  4. Reports **SYSTEM SAFE** with the completion sequence, or **SYSTEM UNSAFE** listing unfinished processes.
- **Resource Request Simulator (`request_resources()`):**
  - Checks whether `Request <= Need` and `Request <= Available`, tentatively updates `Available` and `Allocation`, and runs `safety()` to determine whether the request is **GRANTED** or **DENIED** (with the exact reason).

---

### 5. Disk Head Scheduling (`core/disk.py`)
Simulates disk arm movement across tracks `[0, disk_size - 1]` from an initial `head` position:

| Algorithm | Behavior |
|---|---|
| **FCFS** | Services track requests in exact arrival order |
| **SSTF** | Always moves to the unserviced request with minimum `abs(track - current)` |
| **SCAN** | Sweeps in `direction` (`right` or `left`), servicing requests along the way, reaches the disk boundary (`disk_size - 1` or `0`) if requests remain on the opposite side, then reverses |

- **Outputs:**
  - **Head Trajectory Chart:** Step-by-step track-vs-step line plot with cylinder labels.
  - **Seek Metrics:** Total head movement (tracks), average seek distance per request, step-by-step distance table, and FCFS vs SSTF vs SCAN comparison chart.

---

### 6. Dataset & Export (`core/presets.py`)
- **Curated Presets:**
  - `Lab Standard (Default)` — Balanced benchmark workload with a safe Banker's state.
  - `Idle Gaps & Convoy` — Demonstrates CPU idle gaps and long-burst convoy effects.
  - `Unsafe Deadlock & Fragmentation` — Demonstrates an unsafe Banker's state and rejected memory requests.
- **Custom JSON:** Upload a `.json` file or edit the unified JSON contract directly in the browser with live validation.
- **Exports:** One-click downloads for `analysis.json`, `scheduling.csv`, `memory.csv`, and `disk.csv`.

---

## Unified JSON Input Contract

All four subsystems read from a single JSON structure:

```json
{
  "quantum": 2,
  "processes": [
    {"pid": "P1", "arrival": 0, "burst": 7, "priority": 2},
    {"pid": "P2", "arrival": 2, "burst": 4, "priority": 1}
  ],
  "memory": {
    "blocks": [100, 500, 200, 300, 600],
    "requests": [212, 417, 112, 426, 95]
  },
  "deadlock": {
    "available": [3, 3, 2],
    "allocation": [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]],
    "maximum": [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
  },
  "disk": {
    "requests": [98, 183, 37, 122, 14, 124, 65, 67],
    "head": 53,
    "disk_size": 200,
    "direction": "right"
  }
}
```

---

## Visualization Engine (`utils/charts.py`)

| Function | Chart Type | Purpose |
|---|---|---|
| `make_gantt(segments)` | Horizontal stacked/offset bar | Single-row CPU execution trace with `IDLE` gaps |
| `make_cpu_comparison_chart(cpu_results)` | Grouped bar | Compares Avg Waiting, Turnaround, and Response |
| `make_memory_map(mem_result)` | Horizontal stacked bar | Visualizes request placements and free space per block |
| `make_memory_comparison_chart(mem_results)` | Bar | Compares utilization % across fit strategies |
| `make_disk_head_chart(disk_result)` | Line + markers | Plots head sweep trajectory across tracks |
| `make_disk_comparison_chart(disk_results)` | Bar | Compares total head movement across algorithms |

---

## Automated Testing (`tests/test_core.py`)

Run the unit test suite from `OS_Resource_Studio/`:

```bash
python3 -m unittest discover -s tests -v
```

The test suite verifies:
- `TestScheduler`: FCFS, SJF, Priority, and RR averages + `IDLE` gap insertion and CPU utilization.
- `TestMemory`: First Fit, Best Fit, and Worst Fit allocation counts and block contents.
- `TestDeadlock`: Safe sequence, execution trace, resource request grant/denial, and unsafe state detection.
- `TestDisk`: Total head movement for FCFS, SSTF, and SCAN.
- `TestChartsAndPresets`: End-to-end validation of all presets and Plotly chart generation.
