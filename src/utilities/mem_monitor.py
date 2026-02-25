"""Live memory-usage monitor TUI and memory-profile configuration.

Runs a daemon thread that periodically samples process and system memory,
rendering a compact Rich Live panel at the bottom of the terminal.

Usage:
    from src.utilities.mem_monitor import MemoryMonitor, NullMonitor, mem_profile

    profile = mem_profile(16)  # configure for 16 GB cap

    with MemoryMonitor(limit_gb=16) as mon:
        mon.stage = "EDA"
        ...  # pipeline work
        mon.stage = "Modelling"
        ...

Use NullMonitor() as a silent drop-in replacement when monitoring is disabled.
"""

import os
import sys
import threading
import time
from dataclasses import dataclass

import psutil
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

# On Windows with legacy console (cp1252), Rich's box-drawing characters can
# cause UnicodeEncodeError.  Wrapping stdout in utf-8 avoids this.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _fmt(byte_count):
    """Format bytes as a human-readable string (MB or GB)."""
    gb = byte_count / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{byte_count / (1024 ** 2):.0f} MB"


VALID_MEM_LIMITS = (8, 16, 32, 64)


@dataclass
class MemoryProfile:
    """Concurrency settings derived from a memory budget.

    Attributes
    ----------
    limit_gb : int or None
        Requested cap in GB (None = unlimited).
    sequential : bool
        Force sequential pipeline stages (no ThreadPoolExecutor in pipeline.py).
    max_workers : int
        Max threads/processes for IP geolocation and UA parsing.
    pipeline_workers : int
        Max concurrent stage-1 workers in the pipeline (1 = sequential).
    """

    limit_gb: int | None
    sequential: bool
    max_workers: int
    pipeline_workers: int

    def describe(self) -> str:
        """Return a human-readable summary of the profile settings."""
        cap = f"{self.limit_gb} GB" if self.limit_gb else "unlimited"
        mode = "sequential" if self.sequential else f"{self.pipeline_workers} concurrent stages"
        return (
            f"Memory profile: {cap}  |  "
            f"pipeline={mode}, workers={self.max_workers}"
        )


def mem_profile(limit_gb=None):
    """Build a MemoryProfile for the given GB cap.

    Parameters
    ----------
    limit_gb : int or None
        One of 8, 16, 32, 64, or None (unlimited).

    Returns
    -------
    MemoryProfile
    """
    cpu_workers = max(1, (os.cpu_count() or 4) - 2)

    if limit_gb is None:
        return MemoryProfile(
            limit_gb=None,
            sequential=False,
            max_workers=cpu_workers,
            pipeline_workers=4,
        )

    if limit_gb not in VALID_MEM_LIMITS:
        raise ValueError(
            f"--mem-limit must be one of {VALID_MEM_LIMITS}, got {limit_gb}"
        )

    # 8 GB  — fully sequential, single worker
    # 16 GB — sequential pipeline stages, 2 data workers
    # 32 GB — 2 concurrent stages, half the cpu workers
    # 64 GB — 4 concurrent stages (full parallel), all cpu workers
    profiles = {
        8:  MemoryProfile(limit_gb=8,  sequential=True,  max_workers=1,                    pipeline_workers=1),
        16: MemoryProfile(limit_gb=16, sequential=True,  max_workers=min(2, cpu_workers),   pipeline_workers=1),
        32: MemoryProfile(limit_gb=32, sequential=False, max_workers=min(4, cpu_workers),   pipeline_workers=2),
        64: MemoryProfile(limit_gb=64, sequential=False, max_workers=cpu_workers,           pipeline_workers=4),
    }
    return profiles[limit_gb]


class MemoryMonitor:
    """Background memory monitor with a Rich Live TUI status line.

    Parameters
    ----------
    interval : float
        Sampling interval in seconds (default 2.0).
    limit_gb : int or None
        Memory cap in GB shown in the TUI (display only).
    """

    def __init__(self, interval=2.0, limit_gb=None):
        self.interval = interval
        self.limit_gb = limit_gb
        self.stage = "init"
        self._peak_rss = 0
        self._stop = threading.Event()
        self._thread = None
        self._live = None
        self._console = Console()
        self._process = psutil.Process()
        self._start_time = None
        self._lock = threading.Lock()

    # -- rendering ----------------------------------------------------------

    def _build_table(self):
        """Build a compact Rich Table showing memory stats."""
        mem = self._process.memory_info()
        rss = mem.rss
        with self._lock:
            if rss > self._peak_rss:
                self._peak_rss = rss
            peak = self._peak_rss

        vm = psutil.virtual_memory()
        elapsed = time.time() - self._start_time

        title = "Memory Monitor"
        if self.limit_gb:
            title += f"  (limit: {self.limit_gb} GB)"
        table = Table(
            title=title,
            title_style="bold cyan",
            show_header=True,
            header_style="bold",
            expand=False,
            padding=(0, 1),
        )
        table.add_column("Stage", style="green", min_width=14)
        table.add_column("RSS", style="yellow", justify="right", min_width=10)
        table.add_column("Peak RSS", style="red", justify="right", min_width=10)
        if self.limit_gb:
            table.add_column("Budget", justify="right", min_width=10)
        table.add_column("System", justify="right", min_width=16)
        table.add_column("Elapsed", justify="right", min_width=8)

        # System memory bar:  used / total  (percent)
        sys_text = Text()
        pct = vm.percent
        if pct > 90:
            color = "bold red"
        elif pct > 75:
            color = "yellow"
        else:
            color = "green"
        sys_text.append(f"{_fmt(vm.used)}", style=color)
        sys_text.append(f" / {_fmt(vm.total)}")
        sys_text.append(f"  {pct:.0f}%", style=color)

        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        if hrs:
            elapsed_str = f"{hrs}h{mins:02d}m{secs:02d}s"
        elif mins:
            elapsed_str = f"{mins}m{secs:02d}s"
        else:
            elapsed_str = f"{secs}s"

        row = [self.stage, _fmt(rss), _fmt(peak)]
        if self.limit_gb:
            limit_bytes = self.limit_gb * (1024 ** 3)
            usage_pct = rss / limit_bytes * 100
            budget_text = Text()
            if usage_pct > 90:
                budget_text.append(f"{usage_pct:.0f}%", style="bold red")
            elif usage_pct > 75:
                budget_text.append(f"{usage_pct:.0f}%", style="yellow")
            else:
                budget_text.append(f"{usage_pct:.0f}%", style="green")
            row.append(budget_text)
        row.extend([sys_text, elapsed_str])

        table.add_row(*row)
        return table

    # -- background loop ----------------------------------------------------

    def _run(self):
        """Sampling loop executed in the daemon thread."""
        try:
            with Live(
                self._build_table(),
                console=self._console,
                refresh_per_second=1,
                transient=False,
            ) as live:
                self._live = live
                while not self._stop.is_set():
                    live.update(self._build_table())
                    self._stop.wait(self.interval)
                # Final snapshot
                live.update(self._build_table())
        except Exception:
            pass  # don't crash the pipeline if rendering fails

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        """Start the monitor thread."""
        self._start_time = time.time()
        self._peak_rss = self._process.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the monitor thread and print a final summary."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

        elapsed = time.time() - self._start_time
        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        if hrs:
            elapsed_str = f"{hrs}h{mins:02d}m{secs:02d}s"
        elif mins:
            elapsed_str = f"{mins}m{secs:02d}s"
        else:
            elapsed_str = f"{secs}s"

        self._console.print()
        self._console.rule("[bold cyan]Memory Summary[/bold cyan]")
        self._console.print(f"  Peak RSS:     [bold red]{_fmt(self._peak_rss)}[/bold red]")
        if self.limit_gb:
            limit_bytes = self.limit_gb * (1024 ** 3)
            peak_pct = self._peak_rss / limit_bytes * 100
            self._console.print(f"  Budget used:  {peak_pct:.0f}% of {self.limit_gb} GB")
        self._console.print(f"  Total time:   {elapsed_str}")
        self._console.rule()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stage = "done"
        self.stop()


class NullMonitor:
    """No-op drop-in replacement for MemoryMonitor when monitoring is disabled."""

    stage = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
