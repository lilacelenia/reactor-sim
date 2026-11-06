"""
verify_all.py -- the whole verification suite, independent parts running at the same time.

Every verify script and every reference run is a separate process; comparisons wait for
their runs. The longest tasks start first. verify_live.py checks wall-clock pacing, so it
runs alone at the end. Logs go to results/logs/<task>.log; a summary is printed at the end
and the exit status is non-zero if any task failed.

    python3 verify_all.py              # everything (default: one process per performance core)
    python3 verify_all.py --jobs 4     # at most 4 processes at a time (or REACTOR_JOBS=4)
    python3 verify_all.py --quick      # skip the reference runs and their comparisons
"""
import subprocess
import sys
import time
from pathlib import Path

from parallel import default_jobs, pop_jobs, single_thread_env

PY = sys.executable

# name: (command, estimated minutes on one core, dependencies)
CHECKS = {
    "solver": (["verify_solver.py"], 1.5, []),
    "run_core": (["run_core.py", "mitr2_like.yaml"], 1.5, []),
    "blades": (["verify_blades.py"], 10, []),
    "control": (["verify_control.py"], 12, []),
    "feedback": (["verify_feedback.py"], 3, []),
    "feedback_lof": (["verify_feedback.py", "lof"], 8, []),
    "kinetics": (["verify_kinetics.py", "cube", "steady"], 1, []),
    "startup": (["verify_startup.py"], 25, []),
    "gui": (["gui_smoke_test.py"], 1, []),
}
REFERENCE = {   # script: ({run name: minutes}, comparison arguments)
    "verify_kinetics.py": ({"direct_h": 12, "direct_2h": 7, "direct_4h": 4, "direct_coarse": 1.5,
                            "pcqs_1.0": 0.5, "pcqs_0.25": 1.5, "adiabatic_0.25": 6}, ["compare"]),
    "verify_feedback.py": ({"direct_h": 18, "direct_2h": 10, "direct_4h": 6, "pcqs_1.0": 1,
                            "pcqs_0.25": 2, "adiabatic_0.25": 5, "nofb_pcqs_0.25": 1,
                            "lof_direct_h": 22, "lof_direct_2h": 12, "lof_direct_4h": 6,
                            "lof_pcqs_0.4": 2.5, "lof_pcqs_0.2": 3},
                           ["compare", "compare_lof"]),
    "verify_control.py": ({"direct_h": 12, "direct_2h": 7, "direct_4h": 4, "pcqs_1.0": 1.2,
                           "pcqs_0.5": 2}, ["compare"]),
}
EXCLUSIVE = {"live": (["verify_live.py"], 5, [])}     # wall-clock pacing: run alone


def tasks(quick):
    t = dict(CHECKS)
    if not quick:
        for script, (runs, compare) in REFERENCE.items():
            stem = Path(script).stem.replace("verify_", "")
            names = []
            for run, minutes in runs.items():
                names.append(f"{stem}_run_{run}")
                t[names[-1]] = ([script, "run", run, "--jobs", "1"], minutes, [])
            t[f"{stem}_compare"] = ([script, *compare], 1, names)
    return t


def main():
    args, jobs = pop_jobs(sys.argv[1:])
    jobs = jobs or default_jobs()
    todo = tasks("--quick" in args)
    logs = Path("results/logs")
    logs.mkdir(parents=True, exist_ok=True)
    env = single_thread_env()
    running, done, t0 = {}, {}, time.time()
    total = len(todo) + len(EXCLUSIVE)
    print(f"{total} tasks, up to {jobs} at a time; logs in {logs}/", flush=True)

    def start(name, cmd):
        f = open(logs / f"{name}.log", "w")
        running[name] = (subprocess.Popen([PY, *cmd], stdout=f, stderr=subprocess.STDOUT, env=env),
                         f, time.time())

    def reap():
        for name, (p, f, ts) in list(running.items()):
            if p.poll() is not None:
                f.close()
                done[name] = (p.returncode, time.time() - ts)
                del running[name]
                status = "ok" if p.returncode == 0 else f"FAILED ({p.returncode})"
                print(f"  [{len(done):2d}/{total}] {name:28s} {status:12s} {done[name][1] / 60:5.1f} min",
                      flush=True)

    while todo or running:
        reap()
        ready = sorted((n for n, (_, _, deps) in todo.items() if all(d in done for d in deps)),
                       key=lambda n: -todo[n][1])        # longest first
        for name in ready:
            if len(running) >= jobs:
                break
            cmd, _, deps = todo.pop(name)
            if any(done[d][0] != 0 for d in deps):
                done[name] = (-1, 0.0)
                print(f"  [{len(done):2d}/{total}] {name:28s} skipped (a run it needs failed)")
                continue
            start(name, cmd)
        time.sleep(0.5)
    for name, (cmd, _, _) in EXCLUSIVE.items():
        start(name, cmd)
        while running:
            reap()
            time.sleep(0.5)

    failed = [n for n, (rc, _) in done.items() if rc != 0]
    print(f"\n{len(done) - len(failed)} of {len(done)} tasks passed in {(time.time() - t0) / 60:.1f} min"
          + (f"; failed: {', '.join(failed)} (see results/logs/)" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
