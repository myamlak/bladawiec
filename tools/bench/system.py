#!/usr/bin/env python3
"""Machine-state recording for the comparative benchmark harness: the
power plan, the effective threads, the wall AND the CPU. Windows-native
helpers only; the pyscf side runs in WSL and must not import this module.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import re
import subprocess
import time
from typing import Optional


def ToWslPath(path: pathlib.Path) -> str:
    """Translate a Windows path to the /mnt/... path WSL sees."""
    text = str(path).replace("\\", "/")
    drive, _, rest = text.partition(":")
    return f"/mnt/{drive.lower()}{rest}"


def PhysicalCoreCount() -> int:
    """Physical-core count via GetLogicalProcessorInformation
    (RelationProcessorCore = 0); falls back to logical/2."""
    try:
        import ctypes
        from ctypes import wintypes

        class _SysLogicalProcessorInfo(ctypes.Structure):
            _fields_ = [("mask", wintypes.ULONG_PTR),
                        ("level", wintypes.BYTE),
                        ("relation", wintypes.BYTE),
                        ("reserved", wintypes.BYTE * 2)]

        buffer = ctypes.create_string_buffer(256)
        size = wintypes.DWORD(256)
        func = ctypes.windll.kernel32.GetLogicalProcessorInformation
        func.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        while not func(ctypes.cast(buffer, ctypes.c_void_p),
                       ctypes.byref(size)):
            err = ctypes.get_last_error()
            if err != 87:  # ERROR_INSUFFICIENT_BUFFER
                break
            buffer = ctypes.create_string_buffer(size.value)
        entries = []
        for offset in range(0, size.value, ctypes.sizeof(_SysLogicalProcessorInfo)):
            entry = _SysLogicalProcessorInfo.from_buffer_copy(
                buffer.raw[offset:offset + ctypes.sizeof(_SysLogicalProcessorInfo)])
            entries.append(entry)
        cores = sum(1 for e in entries if e.relation == 0)
        return cores if cores > 0 else max(1, os.cpu_count() // 2)
    except Exception:
        return max(1, os.cpu_count() // 2)


def _OemDecode(data: bytes) -> str:
    """Decode native-console output. powercfg writes in the console
    OEM codepage (cp852 on this host), not the locale encoding: the
    cp1250 decode produced U+FFFD mojibake on 2026-09-02 (and the
    replacement char then crashed the cp1250 console print of the
    machine line). GetOEMCP names the right codec; errors='replace'
    keeps the result annotation-grade when even it cannot map a byte,
    and the caller strips any surviving U+FFFD so a scheme name can
    never kill a pass print (a mojibake name is data, a traceback is
    not)."""
    try:
        import ctypes
        cp = ctypes.windll.kernel32.GetOEMCP()
        codec = f"cp{cp}" if cp else None
    except Exception:
        codec = None
    if codec is None:
        codec = __import__("locale").getpreferredencoding(False)
    return data.decode(codec, errors="replace")


def PowerScheme() -> Optional[str]:
    """The active Windows power scheme, e.g. 'Balanced'. Bytes mode:
    subprocess's text mode decodes
    with the locale encoding, which cannot read powercfg's OEM-codepage
    output on this cp1250 host (a reader-thread UnicodeDecodeError
    traceback sat on every MachineState call)."""
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"],
                             capture_output=True, timeout=15)
        text = _OemDecode(out.stdout).replace("\uFFFD", "?")
        match = re.search(r"\(([^)]+)\)", text)
        return match.group(1) if match else text.strip()[:120]
    except Exception:
        return None


def _MemoryStatusGiB() -> Optional[dict]:
    """The GlobalMemoryStatusEx quantities in GiB, keyed by field name
    (total/avail physical RAM, total/avail page-file commit); None when
    the readout fails (the memory gate then relies on the job cap
    alone)."""
    try:
        import ctypes
        from ctypes import wintypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD),
                        ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", wintypes.ULARGE_INTEGER),
                        ("ullAvailPhys", wintypes.ULARGE_INTEGER),
                        ("ullTotalPageFile", wintypes.ULARGE_INTEGER),
                        ("ullAvailPageFile", wintypes.ULARGE_INTEGER),
                        ("ullTotalVirtual", wintypes.ULARGE_INTEGER),
                        ("ullAvailVirtual", wintypes.ULARGE_INTEGER),
                        ("ullAvailExtendedVirtual", wintypes.ULARGE_INTEGER)]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        func = ctypes.windll.kernel32.GlobalMemoryStatusEx
        func.argtypes = [ctypes.c_void_p]
        func.restype = wintypes.BOOL
        if not func(ctypes.byref(status)):
            return None

        def _GiB(quantity) -> float:
            """A wintypes.ULARGE_INTEGER quantity in GiB. Python 3.13+
            made it a plain int type, not a union with .QuadPart;
            getattr() reads either shape (int() raises on the pre-3.13
            union)."""
            return int(getattr(quantity, "QuadPart", quantity)) / (2 ** 30)

        return {name: _GiB(getattr(status, name))
                for name in ("ullTotalPhys", "ullAvailPhys",
                             "ullTotalPageFile", "ullAvailPageFile")}
    except Exception:
        return None


def _RamGiB() -> Optional[tuple]:
    """(total, free) physical RAM in GiB via GlobalMemoryStatusEx; None
    when the readout fails (the memory gate then relies on the job cap
    alone)."""
    status = _MemoryStatusGiB()
    if status is None:
        return None
    return (status["ullTotalPhys"], status["ullAvailPhys"])


def FreeRamGiB() -> Optional[float]:
    """Host free physical RAM in GiB (the memory-gate floor check)."""
    ram = _RamGiB()
    return ram[1] if ram else None


def TotalRamGiB() -> Optional[float]:
    """Host total physical RAM in GiB (recorded in the machine state)."""
    ram = _RamGiB()
    return ram[0] if ram else None


def AvailCommitGiB() -> Optional[float]:
    """Host available commit (ullAvailPageFile) in GiB - the binding
    resource for cap-class runs (the 2026-08-31 c60 double-death: free
    RAM 20.1 GiB passed the floor while avail commit bottomed at 4.96
    and the child died 0xC0000409). None when the probe fails - the
    avail-commit floor check then fails closed."""
    status = _MemoryStatusGiB()
    if status is None:
        return None
    return status["ullAvailPageFile"]


def CpuName() -> Optional[str]:
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return value.strip()
    except Exception:
        return platform.processor()


# --- the power/thermal/battery state (Amendment 2's covariate list) ----------
# The counter-based readouts (_ProbePowerShell) are unreachable on hosts whose
# performance-counter subsystem reports no counter set, so these read the
# state through the channels that still answer: kernel32 for the AC/battery
# line, WMI for the CPU microcode, the memory speed and the coarse clock.  A
# readout that fails is None WITH its reason recorded by the caller - never a
# number that was not measured.


def _ACLineStatusValue() -> Optional[dict]:
    """GetSystemPowerStatus via ctypes: the AC/battery state. ctypes rather
    than WMI because it is one call with no provider to start, and the flag
    values are the Win32 constants (ACLineStatus 0=offline, 1=online,
    255=unknown; BatteryFlag 128=no battery)."""
    import ctypes
    from ctypes import wintypes

    class _PowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", wintypes.BYTE),
                    ("BatteryFlag", wintypes.BYTE),
                    ("BatteryLifePercent", wintypes.BYTE),
                    ("SystemStatusFlag", wintypes.BYTE),
                    ("BatteryLifeTime", wintypes.DWORD),
                    ("BatteryFullLifeTime", wintypes.DWORD)]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        status = _PowerStatus()
        if not kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
    except Exception:
        return None
    return {"ac_line_status": {
                0: "offline", 1: "online", 255: "unknown"}.get(
                    status.ACLineStatus, f"raw_{status.ACLineStatus}"),
            "battery_percent": (None if status.BatteryLifePercent == 255
                                else status.BatteryLifePercent),
            "no_battery": bool(status.BatteryFlag & 128)}


def _WmiReadout(namespace: str, klass: str, field: str) -> Optional[float]:
    """One WMI property read (a PowerShell CIM query), float-cast, or None
    when the provider does not answer.  One process spawn per call: the
    caller reads these per BLOCK, never per call."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance -Namespace {namespace} -ClassName {klass} | "
             f"Select-Object -First 1 -ExpandProperty {field})"],
            capture_output=True, timeout=30)
        if out.returncode != 0:
            return None
        text = out.stdout.decode("utf-8", errors="replace").strip()
        return float(text) if text else None
    except Exception:
        return None


def Amendment2Covariates() -> dict:
    """The Amendment 2 covariate fields this host can reach, each with an
    explicit reason when it cannot: the AC/battery state, the CPU microcode
    revision, the memory speed and the coarse clock (the registry's nominal
    and WMI's CurrentClockSpeed).

    The per-call frequency SAMPLES and the throttling counters stay
    unreachable (the performance-counter subsystem: Get-Counter -ListSet
    reports no counter set on a host where that is broken), and the caller
    records them as not collectable rather than as a zero.
    """
    readouts = {
        "ac_power": _ACLineStatusValue(),
        "cpu_microcode_revision": _WmiReadout("root/cimv2", "Win32_Processor",
                                              "MicrocodeRevision"),
        "memory_speed_mhz": _WmiReadout("root/cimv2", "Win32_PhysicalMemory",
                                        "Speed"),
        "current_clock_mhz": _WmiReadout("root/cimv2", "Win32_Processor",
                                         "CurrentClockSpeed"),
        "max_clock_mhz": _WmiReadout("root/cimv2", "Win32_Processor",
                                     "MaxClockSpeed"),
    }
    readouts["not_collectable"] = {
        key: reason for key, reason in (
            ("frequency_samples_during_call",
             "no per-call frequency sampling: the performance-counter "
             "subsystem is unreachable on this host (Get-Counter -ListSet "
             "reports no counter set); current_clock_mhz is the coarse WMI "
             "proxy, read once per block"),
            ("throttling_counters",
             "same counter subsystem; no WMI equivalent exists"),
            ("package_core_temperatures",
             "MSAcpi_ThermalZoneTemperature is absent on this host (the "
             "desktop case its probe documents)"),
            ("numa_policy",
             "single-node consumer machine; the [resources] thread_cap is "
             "the only placement knob and it is recorded per call"),
            ("cpu_microcode_revision",
             "Win32_Processor.MicrocodeRevision did not answer on this "
             "host (the provider as queried returns nothing) - recorded as "
             "null WITH this reason rather than as a missing field"),
            ("memory_speed_mhz",
             "Win32_PhysicalMemory.Speed did not answer on this host"),
            ("current_clock_mhz",
             "Win32_Processor.CurrentClockSpeed did not answer on this "
             "host; no per-call frequency sample is available either"),
            ("max_clock_mhz",
             "Win32_Processor.MaxClockSpeed did not answer on this host"),
        ) if readouts.get(key) is None}
    return readouts


def MachineState() -> dict:
    """The machine-state block recorded once per run."""
    logical = os.cpu_count() or 0
    physical = PhysicalCoreCount()
    return {
        "date_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_name": CpuName(),
        "cpu_logical_cores": logical,
        "cpu_physical_cores": physical,
        "power_scheme": PowerScheme(),
        "hyperthreaded": logical > physical,
        "total_ram_gib": TotalRamGiB(),
        "free_ram_gib": FreeRamGiB(),
    }


# --- the per-cell machine-state covariates -----------------------------------
# The calibration cells' run records carry the machine-state
# covariates the fit's block/covariate terms consume (the CPU
# frequency, the temperature, the load).
# The block keys follow the drift layer's machine-state vocabulary
# (cost_table_drift._MACHINE_STATE_KEYS) so a fold consumes them
# unchanged, with two additions the per-cell record needs:
#
#     cpu_frequency_ghz   current CPU frequency in GHz (best-effort
#                         probe; None when the readout fails)
#     temperature_c       CPU package temperature in Celsius
#                         (best-effort; None when the machine exposes
#                         no thermal zone - common on desktops)
#     load_fraction       the host busy fraction over the probe's
#                         ~1 s sampling window, in [0, 1] (a load
#                         average needs a minute-long sample - too
#                         slow for a per-cell probe; the honest
#                         short-window busy fraction is the covariate
#                         the run model consumes)
#     swap_free_kib       the available page-file commit in KiB
#     date_utc            the probe time (the same ISO-UTC shape as
#                         the MachineState block)
#
# Every readout is a best-effort probe: a failure yields None for that
# field, never an exception (a failed probe is recorded honestly, and
# the fit's covariate path drops the None column with a note).  The
# readouts come from ONE powershell invocation (the performance
# counters for frequency/load, WMI for the temperature) plus a ctypes
# page-file read; the probe takes ~1-4 s and is called once per cell
# by the measurement runner (before the cell's first rep, after the
# last - the runner's business; this module only supplies the probe).
COVARIATE_KEYS = ("cpu_frequency_ghz", "temperature_c", "load_fraction",
                  "swap_free_kib", "date_utc")


def _ProbePowerShell() -> Optional[dict]:
    """One powershell invocation reading the performance counters and
    the thermal zone: {"cpu_frequency_ghz", "temperature_c",
    "load_fraction"} or None when powershell itself fails.  The
    frequency readout is the \"Processor Information(_Total) percent
    of maximum frequency\" counter (a measured current-frequency
    fraction, unlike the registry's nominal ~MHz) times the nominal
    frequency; the load readout is the \"Processor(_Total) percent
    processor time\" counter (busy fraction over the counter's ~1 s
    sample window); the temperature is the WMI thermal-zone readout,
    absent on most desktops (then None)."""
    script = (
        "$ErrorActionPreference = 'SilentlyContinue';"
        "$pct = (Get-Counter '\\Processor Information(_Total)\\% of "
        "Maximum Frequency' -SampleInterval 1 -MaxSamples 2 | "
        "Select-Object -Last 1).CounterSamples[0].CookedValue;"
        "$busy = (Get-Counter '\\Processor(_Total)\\% Processor Time' "
        "-SampleInterval 1 -MaxSamples 2 | Select-Object -Last 1)"
        ".CounterSamples[0].CookedValue;"
        "$t = Get-CimInstance MSAcpi_ThermalZoneTemperature "
        "-ErrorAction SilentlyContinue | "
        "Select-Object -First 1 -ExpandProperty CurrentTemperature;"
        "@{frequency_pct=$pct; busy=$busy; thermal_kelvin=$t} | "
        "ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              script],
                             capture_output=True, timeout=30)
        if out.returncode != 0:
            return None
        text = out.stdout.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        values = json.loads(text)
    except Exception:
        return None
    readouts = {}
    nominal = _NominalFrequencyMhz()
    fraction = values.get("frequency_pct")
    readouts["cpu_frequency_ghz"] = (
        round(nominal * float(fraction) / 100.0 / 1000.0, 4)
        if isinstance(fraction, (int, float)) and fraction > 0.0
        and nominal is not None else None)
    busy = values.get("busy")
    readouts["load_fraction"] = (
        round(max(0.0, min(1.0, float(busy) / 100.0)), 4)
        if isinstance(busy, (int, float)) else None)
    kelvin = values.get("thermal_kelvin")
    readouts["temperature_c"] = (
        round(float(kelvin) - 273.15, 1)
        if isinstance(kelvin, (int, float)) and kelvin > 100.0 else None)
    return readouts


def _NominalFrequencyMhz() -> Optional[float]:
    """The nominal CPU frequency in MHz (the registry's ~MHz value);
    None when the readout fails.  The frequency probe multiplies this
    by the measured percent-of-maximum counter."""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        value, _ = winreg.QueryValueEx(key, "~MHz")
        return float(value)
    except Exception:
        return None


def _SwapFreeKibLive() -> Optional[float]:
    """The available page-file commit in KiB (ullAvailPageFile); None
    when the readout fails."""
    status = _MemoryStatusGiB()
    if status is None:
        return None
    return round(status["ullAvailPageFile"] * (2 ** 20), 3)


def MachineCovariates(readouts: dict | None = None) -> dict:
    """The per-cell machine-state covariate block (see COVARIATE_KEYS).

    ``readouts`` (tests and fakes): a dict of live-style readouts
    (cpu_frequency_ghz/temperature_c/load_fraction) - None probes the
    machine live.  Every field is Optional: a failed readout is
    recorded as None, never a raised exception and never a
    fabricated number.  A malformed readouts dict (wrong types,
    out-of-range load) refuses - a caller bug."""
    if readouts is None:
        readouts = _ProbePowerShell() or {}
    else:
        if not isinstance(readouts, dict):
            raise ValueError(f"readouts: expected a dict or None, got "
                             f"{type(readouts).__name__}")
        unknown = sorted(set(readouts) - {"cpu_frequency_ghz",
                                          "temperature_c", "load_fraction"})
        if unknown:
            raise ValueError(f"readouts: unknown key(s) {unknown} - the "
                             f"covariate vocabulary is "
                             f"{COVARIATE_KEYS[:-1]}")
        for key in ("cpu_frequency_ghz", "temperature_c", "load_fraction"):
            value = readouts.get(key)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value,
                                                        (int, float))):
                raise ValueError(f"readouts.{key}: expected a number or "
                                 f"None, got {value!r}")
        load = readouts.get("load_fraction")
        if load is not None and not 0.0 <= load <= 1.0:
            raise ValueError(f"readouts.load_fraction: expected a busy "
                             f"fraction in [0, 1], got {load!r}")
    return {"cpu_frequency_ghz": readouts.get("cpu_frequency_ghz"),
            "temperature_c": readouts.get("temperature_c"),
            "load_fraction": readouts.get("load_fraction"),
            "swap_free_kib": _SwapFreeKibLive(),
            "date_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(
                    timespec="seconds")}


# --- the per-process CPU probe (the quiet-window gate) -----------------------

# The timed-limb clean-rep rule (the 2026-09-10 timing-limb amendment) is
# stated per PROCESS -
# "no other process above 5% CPU" - and the host busy fraction above cannot
# answer it: a busy fraction under the ceiling still hides one pegged core
# on a many-core host.  This probe reads the per-process CPU time directly,
# so a caller can name the process that disqualifies the window instead of
# only knowing the host was warm.
_BUSIEST_PROCESS_LIMIT = 3

# The probe reads GetProcessTimes deltas over its own sampling window
# rather than the "\Process(*)\% Processor Time" performance counter:
# on the 2026-09-12 staging machine the WHOLE counter subsystem is
# unreachable (Get-Counter -ListSet Process reports no counter set on
# the host - the same condition that nulls _ProbePowerShell's
# frequency/load covariates here), and a counter-based probe that
# swallowed that failure would report "no busy process" on a machine
# where it could not measure anything at all - a fabricated quiet
# verdict.  GetProcessTimes needs no counter registry: it is one
# OpenProcess + GetProcessTimes per PID (PROCESS_QUERY_LIMITED_INFORMATION
# is enough for both that and the image name).
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_FILETIME_TICKS_PER_SECOND = 10_000_000


def _ProcessCpuSample() -> Optional[dict]:
    """One process-CPU census: ``{pid: {"name", "cpu_ticks"}}`` with
    ``cpu_ticks`` the process's kernel+user time in FILETIME ticks
    (100 ns), or None when the census itself fails (EnumProcesses
    refuses).  A process that cannot be opened - another user's, or one
    that exited mid-census - is skipped, never zero-filled in."""
    import ctypes
    from ctypes import wintypes

    try:
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except OSError:
        return None

    capacity = 4096
    while True:
        array = (wintypes.DWORD * capacity)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcesses(array, ctypes.sizeof(array),
                                   ctypes.byref(needed)):
            return None
        count = needed.value // ctypes.sizeof(wintypes.DWORD)
        if count < capacity:
            break
        if capacity > 1 << 20:
            return None
        capacity *= 4

    census = {}
    for pid in list(array[:count]):
        if pid == 0:
            continue
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION,
                                      False, int(pid))
        if not handle:
            continue
        try:
            creation = wintypes.FILETIME()
            exited = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(handle, ctypes.byref(creation),
                                            ctypes.byref(exited),
                                            ctypes.byref(kernel),
                                            ctypes.byref(user)):
                continue
            ticks = (((kernel.dwHighDateTime << 32) | kernel.dwLowDateTime) +
                     ((user.dwHighDateTime << 32) | user.dwLowDateTime))
            name = ""
            buffer = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buffer))
            if kernel32.QueryFullProcessImageNameW(handle, 0, buffer,
                                                   ctypes.byref(size)):
                name = pathlib.Path(buffer.value).name
            census[int(pid)] = {"name": name, "cpu_ticks": ticks}
        finally:
            kernel32.CloseHandle(handle)
    return census


def BusiestProcesses(window_seconds: float = 1.0,
                     limit: int = _BUSIEST_PROCESS_LIMIT) -> Optional[list]:
    """The busiest processes by CPU over a ``window_seconds`` window,
    descending, each ``{"name", "pid", "cpu_fraction"}`` - or None when
    the census itself cannot run.

    ``cpu_fraction`` is normalized to ONE core (a fully busy single
    thread reads 1.0), the convention the timed-limb rule's "no other
    process above 5% CPU" is stated in, so a caller compares it against
    a per-core ceiling directly.  A census that runs and finds nothing
    busy returns an empty list - the probe's own measurement - while a
    census that cannot run returns None: "could not measure" and
    "measured nothing" must never collapse into the same value."""
    first = _ProcessCpuSample()
    if first is None:
        return None
    started = time.perf_counter()
    time.sleep(max(0.05, window_seconds))
    elapsed = time.perf_counter() - started
    second = _ProcessCpuSample()
    if second is None:
        return None
    budget = elapsed * _FILETIME_TICKS_PER_SECOND
    if budget <= 0.0:
        return None
    rows = []
    for pid, entry in second.items():
        before = first.get(pid)
        if before is None:
            continue
        fraction = (entry["cpu_ticks"] - before["cpu_ticks"]) / budget
        if fraction <= 0.0:
            continue
        rows.append({"name": entry["name"], "pid": pid,
                     "cpu_fraction": round(fraction, 4)})
    rows.sort(key=lambda row: row["cpu_fraction"], reverse=True)
    return rows[:limit]


