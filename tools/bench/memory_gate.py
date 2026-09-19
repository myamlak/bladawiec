#!/usr/bin/env python3
"""The benchmark memory gate (the 2026-08-29 machine-kill rule).

Hard rule (set 2026-08-29): no benchmark invocation may
exceed 16 GiB of RAM. Two mechanisms enforce it:

1. Hard process cap - every qcx child is launched inside a Windows job
   object with JOB_OBJECT_LIMIT_PROCESS_MEMORY at the cap: the OS kills
   the process the moment it would exceed, instead of the machine dying.
2. Free-RAM floor - a qcx child is not launched unless the host has at
   least the floor free. The 2026-08-29 kill mode was pagefile thrash
   (free RAM 1.6 GiB, pagefile peak 65 GiB), not a single allocation.

Both are fail-closed: if the cap cannot be applied, the child is killed
and the launch refused. The cap covers the qcx side; the pyscf side runs
inside WSL and is bounded by fixture selection instead (per-element aux,
<= 1.6 GiB modeled after the C60/def2-SVP exclusion).

3. Avail-commit floor (opt-in) - a caller that passes an explicit floor
   (run_comparative.py --avail-commit-floor-gib) refuses to launch a
   qcx child unless the host has at least the floor of available commit
   (ullAvailPageFile; the 2026-08-31 c60 double-death: free RAM passed
   while avail commit bottomed out and the child died 0xC0000409).
   CheckAvailCommitFloor FAILS CLOSED - a None probe readout refuses
   (pass-open probes are how the 12 GiB floor got bypassed). Legacy
   call sites without the explicit floor keep the old behavior: only
   the free-RAM check runs, and it passes open on a None readout with
   the job cap still holding on its own.
"""

from __future__ import annotations

import subprocess
from typing import Optional

# The 2026-08-29 user rule and its free-RAM floor.
DefaultCapGiB = 16.0
DefaultFreeFloorGiB = 12.0

_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def RijTensorGiB(n_basis: int, n_aux: int) -> float:
    """The dense-tensor model: 8 bytes per double, n^2 x nAux, in
    GiB. The pre-flight estimate for every ri_j cell."""
    return 8.0 * n_basis * n_basis * n_aux / (2 ** 30)


def CheckFreeRam(free_gib, floor_gib: float) -> bool:
    """True when a child may launch: the host has at least floor_gib
    free. A None readout (the status call failed) does not block - the
    job cap still holds on its own."""
    return free_gib is None or free_gib >= floor_gib


def CheckAvailCommitFloor(avail_gib, floor_gib: float) -> bool:
    """True when a child may launch under the avail-commit floor: the
    host has at least floor_gib of available commit (ullAvailPageFile -
    the binding resource for cap-class runs, the 2026-08-31 c60 double
    death). FAILS CLOSED: a None readout (the probe failed) refuses -
    pass-open probes are how the 12 GiB floor got bypassed. Only call
    sites that pass an explicit floor are gated (run_comparative.py
    --avail-commit-floor-gib); legacy call sites keep the
    CheckFreeRam-only behavior."""
    return avail_gib is not None and avail_gib >= floor_gib


def ApplyToChild(proc: subprocess.Popen, cap_gib: float) -> None:
    """Attach a live child to a fresh job object with a hard
    ProcessMemoryLimit at cap_gib. Fail-closed: when the cap cannot be
    applied the child is killed and RuntimeError raised - an uncapped
    benchmark must never run. The assignment window (process start to
    this call) is the qcx parse phase, far below the cap, so no pre-cap
    allocation can exceed it in practice."""
    try:
        _AttachJobLimit(proc, cap_gib)
    except Exception as exc:
        proc.kill()
        proc.communicate()
        raise RuntimeError(str(exc)) from exc


def _JobLimitInfoClass():
    """The JobObjectExtendedLimitInformation ctypes structure type.

    One definition shared by the cap application (_AttachJobLimit) and
    the after-exit peak read-out (JobPeakMemoryKib): the two must
    always describe the same layout. ctypes stays a function-local
    import so the module remains importable off-Windows (the pyscf/WSL
    side and the Linux CI leg import tools/bench modules).
    """
    import ctypes
    from ctypes import wintypes

    class _JobobjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_ulonglong),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobobjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobobjectBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return _JobobjectExtendedLimitInformation


def JobPeakMemoryKib(proc: subprocess.Popen) -> Optional[int]:
    """Peak process memory of a finished child, in KiB, read back from
    the job object the gate attached (QueryInformationJobObject,
    JobObjectExtendedLimitInformation.PeakProcessMemoryUsed).
    Fail-open: None when the job handle is absent or the read-out
    fails - the read-out is measurement for the memory-guard layers,
    while the cap itself stays the fail-closed enforcer. Read after
    the child has exited (the peak is final then)."""
    handle = getattr(proc, "_memory_job_handle", None)
    if handle is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        info = _JobLimitInfoClass()()
        returned = wintypes.DWORD(0)
        if not kernel32.QueryInformationJobObject(
                handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info),
                ctypes.byref(returned)):
            return None
        return int(info.PeakProcessMemoryUsed) // 1024
    except Exception:
        return None


def _AttachJobLimit(proc: subprocess.Popen, cap_gib: float) -> None:
    import ctypes
    from ctypes import wintypes

    _JobobjectExtendedLimitInformation = _JobLimitInfoClass()

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    hjob = kernel32.CreateJobObjectW(None, None)
    if not hjob:
        raise RuntimeError(f"CreateJobObjectW failed (error "
                           f"{ctypes.get_last_error()})")

    info = _JobobjectExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = (
        _JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    info.ProcessMemoryLimit = int(cap_gib * (2 ** 30))
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    if not kernel32.SetInformationJobObject(
            hjob, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info), ctypes.sizeof(info)):
        raise RuntimeError(f"SetInformationJobObject failed (error "
                           f"{ctypes.get_last_error()})")

    kernel32.AssignProcessToJobObject.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    if not kernel32.AssignProcessToJobObject(
            hjob, wintypes.HANDLE(proc._handle)):
        raise RuntimeError(f"AssignProcessToJobObject failed (error "
                           f"{ctypes.get_last_error()}")

    # The handle must outlive the child: KILL_ON_JOB_CLOSE kills the
    # job's processes when the handle closes. Keep it on the Popen.
    proc._memory_job_handle = hjob
