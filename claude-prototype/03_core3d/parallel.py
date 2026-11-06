"""
parallel.py -- run independent cases (reference transients, whole verify scripts) at the same
time, one process per case.

The solvers are single-threaded (scipy sparse matrix-vector products), so independent runs
scale with the number of cores. On Apple Silicon only the performance cores are counted by
default: an efficiency core runs these solvers several times slower, and a pool sized to
all cores would leave its slowest jobs on them. Each worker is pinned to one BLAS/OpenMP
thread (Accelerate, OpenBLAS, MKL) so processes do not oversubscribe the cores.

    from parallel import run_parallel, default_jobs
    run_parallel(func, ["a", "b", "c"])      # func(name) in separate processes
"""
import multiprocessing as mp
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def performance_cores():
    """Number of performance cores (Apple Silicon: hw.perflevel0), else all logical CPUs."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                                 capture_output=True, text=True, timeout=5)
            n = int(out.stdout.strip())
            if n > 0:
                return n
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def default_jobs():
    """REACTOR_JOBS from the environment if set, else the number of performance cores."""
    env = os.environ.get("REACTOR_JOBS")
    return max(1, int(env)) if env else performance_cores()


def single_thread_env(env=None):
    """Copy of the environment with BLAS/OpenMP limited to one thread per process."""
    env = dict(os.environ if env is None else env)
    for v in THREAD_VARS:
        env[v] = "1"
    return env


def pop_jobs(args):
    """Remove '--jobs N' / '-j N' from an argument list; returns (args, jobs or None)."""
    args, jobs = list(args), None
    for flag in ("--jobs", "-j"):
        if flag in args:
            i = args.index(flag)
            jobs = int(args[i + 1])
            del args[i:i + 2]
    return args, jobs


def run_parallel(func, items, jobs=None):
    """Call func(item) for every item, each in its own process (at most `jobs` at a time),
    and return the results in the order of `items`. func must be a module-level function.
    With one job (or one item) everything runs in this process, as before."""
    items = list(items)
    jobs = min(jobs or default_jobs(), len(items)) if items else 1
    if jobs <= 1:
        return [func(it) for it in items]
    for v in THREAD_VARS:                      # inherited by the spawned workers
        os.environ[v] = "1"
    ctx = mp.get_context("spawn")              # the macOS default; also safe with threads
    results = [None] * len(items)
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        futures = {pool.submit(func, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()  # re-raises a worker's exception here
    return results
